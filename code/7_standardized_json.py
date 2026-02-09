import os
import json
import csv
from pathlib import Path
import random
from tqdm import tqdm


def load_mapping(mapping_file):
    mapping = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) >= 2 and row[0].strip():
                entity = row[0].strip()
                standardized = row[1].strip()
                mapping[entity] = standardized
    
    print(f"✅ Loaded mapping table: {len(mapping)} mapping records")
    return mapping


def normalize_nodes(json_dir, mapping):
    total_files = 0
    modified_files = 0
    total_replacements = 0
    
    json_files = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(Path(root) / file)
    
    print(f"\n📂 Found {len(json_files)} JSON files")
    
    for file_path in tqdm(json_files, desc="Standardizing nodes", unit="file"):
        total_files += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"\n⚠️ Warning: Unable to parse file {file_path}, skipped")
            continue
        
        modified = False
        file_replacements = 0
        
        for item in data:
            if 'start_node' in item:
                original_start = item['start_node'].strip()
                standardized_start = mapping.get(original_start, original_start)
                if standardized_start != original_start:
                    item['start_node'] = standardized_start
                    modified = True
                    file_replacements += 1
            
            if 'end_node' in item:
                original_end = item['end_node'].strip()
                standardized_end = mapping.get(original_end, original_end)
                if standardized_end != original_end:
                    item['end_node'] = standardized_end
                    modified = True
                    file_replacements += 1
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            modified_files += 1
            total_replacements += file_replacements
    
    print(f"\n📊 Standardization Statistics:")
    print(f"  - Files processed: {total_files}")
    print(f"  - Files modified: {modified_files}")
    print(f"  - Total replacements: {total_replacements}")
    print(f"  - Modification rate: {modified_files/total_files*100:.2f}%")


def sample_and_check(json_dir, mapping, sample_ratio=0.01):
    all_files = []
    for root, _, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                all_files.append(Path(root) / file)
    
    sample_size = max(10, int(len(all_files) * sample_ratio))
    if not all_files:
        print("\n⚠️ No JSON files found to check")
        return
    
    selected_files = random.sample(all_files, min(sample_size, len(all_files)))
    
    print(f"\n🔍 Randomly checking {len(selected_files)} files:")
    
    for file_path in selected_files[:5]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data:
                first_item = data[0]
                print(f"  ✓ {file_path.name}: {first_item.get('start_node', 'N/A')} → {first_item.get('relationship', 'N/A')} → {first_item.get('end_node', 'N/A')}")
        except Exception as e:
            print(f"  ✗ Error checking file {file_path}: {str(e)}")


if __name__ == '__main__':
    print("=" * 60)
    print("Step 7: Node Standardization Application")
    print("=" * 60)
    
    mapping_file = 'mapping.csv'
    enhanced_json_dir = './enhanced_json'
    
    if not os.path.exists(mapping_file):
        print(f"❌ Error: Mapping file {mapping_file} does not exist")
        exit(1)
    
    if not os.path.exists(enhanced_json_dir):
        print(f"❌ Error: JSON directory {enhanced_json_dir} does not exist")
        exit(1)
    
    node_mapping = load_mapping(mapping_file)
    
    normalize_nodes(enhanced_json_dir, node_mapping)
    
    sample_and_check(enhanced_json_dir, node_mapping)
    
    print("\n✅ Processing complete!")