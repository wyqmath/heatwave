import csv
import json
from openai import OpenAI
import multiprocessing
import yaml
from pathlib import Path
import time

def load_config(config_path: str = "config/default.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"⚠️ Unable to load configuration file {config_path}: {e}")
        return {
            "llm": {
                "api_key": os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE'),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3-max"
            },
            "files": {
                "output_relation_nodes": "output_relation_nodes.csv",
                "cleaned_nodes": "cleaned_nodes.csv",
                "mapping_csv": "mapping.csv"
            }
        }

config = load_config()
llm_config = config.get("llm", {})
files_config = config.get("files", {})

client = OpenAI(
    api_key=llm_config.get("api_key"),
    base_url=llm_config.get("base_url"),
)

def clean_node_groups(input_file: str, output_file: str):
    print("\n" + "="*60)
    print("Step 1: Clean Node Group Data")
    print("="*60)

    total_groups = 0
    filtered_groups = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        header = next(infile)
        outfile.write(header)

        for line in infile:
            total_groups += 1
            if ',' in line:
                outfile.write(line)
                filtered_groups += 1

    print(f"✅ Cleaning completed!")
    print(f"   Total groups: {total_groups}")
    print(f"   Retained groups: {filtered_groups} (Multi-node groups)")
    print(f"   Filtered groups: {total_groups - filtered_groups} (Single-node groups)")
    print(f"   Output file: {output_file}")

    return filtered_groups

def process_group(args):
    group_idx, entities = args

    try:
        messages = [
            {"role": "system", "content": "You are a professional entity normalization assistant. Your task is to provide a standardized name that best represents a group of related entities. Please answer in English and only return the standardized name in JSON format without any explanations or additional information."},
            {"role": "user", "content": f"""
Please analyze the following list of related entities and provide a standardized name that best represents this group (in English):

{', '.join(entities)}

Return the result strictly in the following JSON format without any explanations or additional information:
{{
    "standardized_name": "standardized name in English"
}}
        """}
        ]

        completion = client.chat.completions.create(
            model=llm_config.get("model", "qwen3-max"),
            messages=messages,
            response_format={"type": "json_object"},
        )

        response_content = completion.choices[0].message.content
        response_json = json.loads(response_content)
        standardized_name = response_json.get("standardized_name", f"Group_{group_idx}")

        print(f"Entity group [{', '.join(entities)}] standardized to '{standardized_name}'")
        return [(entity, standardized_name) for entity in entities]

    except Exception as e:
        print(f"Error processing group {group_idx+1}: {str(e)}")
        return [(entity, f"Group_{group_idx}") for entity in entities]


def main():
    print("\n" + "="*60)
    print("Step 6: Node Group Cleaning + LLM Standardization Mapping")
    print("="*60)

    input_file = files_config.get("output_relation_nodes", "output_relation_nodes.csv")
    cleaned_file = files_config.get("cleaned_nodes", "cleaned_nodes.csv")
    output_file = files_config.get("mapping_csv", "mapping.csv")

    if not Path(input_file).exists():
        print(f"❌ Error: Input file does not exist: {input_file}")
        print(f"   Please run Step 5 (5_nodes_group.py) first to generate the node group file")
        return

    start_time = time.time()
    filtered_groups = clean_node_groups(input_file, cleaned_file)

    if filtered_groups == 0:
        print(f"\n⚠️ Warning: No multi-node groups to standardize")
        return

    print("\n" + "="*60)
    print("Step 2: LLM Node Standardization Mapping")
    print("="*60)

    groups = []
    with open(cleaned_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                entities = line.split(',')
                entities = [entity.strip() for entity in entities if entity.strip()]
                if entities:
                    groups.append(entities)

    print(f"📊 Loaded {len(groups)} node groups")

    manager = multiprocessing.Manager()
    mapping_data = manager.list()

    num_processes = min(4, multiprocessing.cpu_count())
    print(f"🚀 Using {num_processes} processes for parallel processing")

    tasks = [(idx, entities) for idx, entities in enumerate(groups)]

    print(f"\nProcessing started...")
    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.map(process_group, tasks)

    for result in results:
        mapping_data.extend(result)

    pool.close()
    pool.join()

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['entity', 'standardized_name'])
        writer.writerows(mapping_data)

    elapsed_time = time.time() - start_time
    print("\n" + "="*60)
    print("✅ Processing completed!")
    print("="*60)
    print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    print(f"📁 Output files:")
    print(f"   - {cleaned_file} ({filtered_groups} groups)")
    print(f"   - {output_file} ({len(mapping_data)} mappings)")
    print("="*60)


if __name__ == '__main__':
    main()