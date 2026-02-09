import json
import os
from openai import OpenAI
import re
import time
from concurrent.futures import ThreadPoolExecutor
import threading

def call_with_messages(i, content):
    messages = [{'role': 'system',
                 'content': """As an expert in heatwave disaster analysis, extract comprehensive entity relationships from research texts. Follow these revised guidelines:

# Mandatory Output Format
• STRICTLY USE JSON format with ALL relationships in an array
• Each relationship MUST be a JSON object with EXACTLY 3 fields:
  {
    "start_node": "PascalCaseString", 
    "relationship": "activeVerbPhrase",
    "end_node": "PascalCaseString"
  }
• ABSOLUTELY NO other formats allowed (no plain text, no markdown tables)
• If JSON parsing fails, relationships will be LOST

# Relationship Type Requirements
1. Relational: Focus on specific connections between two concepts
   ("HighTemperature correlatesWith MortalityRisk")
2. Causal: Show direct cause-effect relationships  
   ("ClimateChange causes HeatwaveIntensity")
3. Effect: Demonstrate action-influence relationships
   ("CoolingCenters mitigate HeatStress")

# Enhanced Quantity Enforcement
1. Minimum Relationship Requirements:
   - ABSOLUTELY MUST EXTRACT 8-15 relationships
   - If text is brief, use domain knowledge to add IMPLICIT relationships marked with (*)
   - Never output fewer than 8 relationships

2. New Relationship Expansion Techniques:
   • Vertical Chaining: Expand single relationships into chains 
     (e.g., "A→B→C" becomes "A causes B" and "B leads to C")
   • Cross-Dimension Links: Connect entities across different categories
     (e.g., "HeatWarningSystem reduces HealthcareCosts")

3. Mandatory Self-Check Before Output:
   - Count relationships: MUST have 8-15
   - If <8: Add implicit relationships using:
     1. Reverse relationships (e.g., "X mitigatedBy Y")
     2. Synonym-based relationships
     3. Causal chains from existing nodes

# Revised Examples (Showcasing Quantity):
[
  {"start_node":"ClimateChange", "relationship":"intensifies", "end_node":"HeatwaveDuration"},
  {"start_node":"NighttimeCooling", "relationship":"reduces", "end_node":"HeatRelatedMortality"},
  {"start_node":"UrbanPlanning", "relationship":"prioritizes", "end_node":"ShadeInfrastructure"},
  {"start_node":"HeatwaveEarlyWarning", "relationship":"triggers", "end_node":"EmergencyProtocols"},
  {"start_node":"VulnerablePopulations", "relationship":"require", "end_node":"TargetedOutreach"},
  {"start_node":"ExtremeHeat", "relationship":"accelerates", "end_node":"InfrastructureAging"},
  {"start_node":"WorkplaceRegulations", "relationship":"mandate", "end_node":"HydrationBreaks"}, 
  {"start_node":"HeatStress", "relationship":"correlatesWith", "end_node":"CognitiveDecline"},
  {"start_node":"CommunityNetworks", "relationship":"facilitate", "end_node":"ElderlyCheckIns"}
]

# Critical Enforcement Additions:
• STRICT PENALTY: If output has <8 relationships, 20% credit deduction
• FORMAT REQUIREMENT: Non-JSON output will cause complete rejection
• REQUIRED PATTERNS:
  - At least 1 multi-hop relationship (A→B→C)
  - Minimum 2 cross-category relationships
• FORMAT LENIENCY:
  - Temporary list allowed if JSON parsing fails:
    ["HighTemperature→increases→MortalityRisk", ...]
"""
                },
                {'role': 'user', 'content': f"""{content}"""}]

    try:
        from agents.base_agent import ConfigManager
        config = ConfigManager.load_config()
        api_key = config.get('llm', {}).get('api_key', os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE'))
        base_url = config.get('llm', {}).get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        model = config.get('llm', {}).get('model', 'qwen3-max')
    except:
        api_key = os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE')
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = "qwen3-max"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    content_str = completion.choices[0].message.content

    try:
        parsed_content = json.loads(content_str)
    except json.JSONDecodeError as e:
        with open(f'error_{i}_raw.txt', 'w') as f:
            f.write(content_str)
        
        repaired = content_str
        repaired = re.sub(r"(?<!\\)'", '"', repaired)
        repaired = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', repaired)
        repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
        
        try:
            parsed_content = json.loads(repaired)
        except json.JSONDecodeError:
            from ast import literal_eval
            try:
                parsed_content = literal_eval(repaired)
            except:
                raise ValueError(f"Unparsable JSON content: {content_str[:200]}... (Full content saved to error_{i}_raw.txt)")

    if isinstance(parsed_content, list):
        parsed_content = [
            json.loads(item) if isinstance(item, str) and item.strip().startswith('{') else item
            for item in parsed_content
        ]
        parsed_content = [item for item in parsed_content if isinstance(item, dict)]
    elif isinstance(parsed_content, dict):
        pass
    else:
        pass

    filename = f'{str(i)}.json'
    
    os.makedirs('json', exist_ok=True)
    with open(f"json/{filename}", 'w', encoding='utf-8') as f:
        json.dump(parsed_content, f, ensure_ascii=False, indent=4)

def process_document(args):
    i, doc = args
    try:
        start_time = time.time()
        content = []
        ti_content = ""
        ut_value = ""
        
        if "UT " in doc:
            ut_part = doc.split("UT ")[1]
            ut_value = ut_part.split('\n')[0].strip().split()[-1]
        
        ti_match = re.search(r'TI (.*?)(?=\n[A-Z]{2} )', doc, re.DOTALL)
        if ti_match:
            ti_content = re.sub(r'\s+', ' ', ti_match.group(1).replace('\n', ' ')).strip()
            content.append(ti_content)
        
        ab_match = re.search(r'AB (.*?)(?=\n[A-Z]{2} )', doc, re.DOTALL)
        if ab_match:
            ab_content = ab_match.group(1).replace('\n', ' ').strip()
            content.append(ab_content)
        
        final_content = " ".join(content)
        if not final_content:
            raise ValueError("No valid content found (missing TI and AB fields)")
        
        call_with_messages(i=i, content=final_content)
        elapsed = time.time() - start_time
        print(f"Document {i} processed, time elapsed {elapsed:.2f} seconds")
        return True
    except Exception as e:
        with lock:
            error_file.write(f"{i}Error: {str(e)}\n")
            print(f"{i}Error: {str(e)}")
        return False

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    try:
        from agents.base_agent import ConfigManager
        config = ConfigManager.load_config()
        input_file = config.get('data_processing', {}).get('input_file', 'paper.txt')
        output_dir = config.get('data_processing', {}).get('output_dir', 'json/')
        max_workers = config.get('data_processing', {}).get('max_workers', 15)
    except:
        input_file = 'paper.txt'
        output_dir = 'json/'
        max_workers = 15

    os.makedirs(output_dir, exist_ok=True)

    type_organic = "HeatAdaptation"
    error_file = open(f"{type_organic}_errorfile.txt", mode="a", newline="\n")
    lock = threading.Lock()

    print(f"📄 Reading input file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as file:
        raw_content = file.read()
        
        documents = re.split(r'\n(?=PT )', raw_content)
        
        valid_documents = [doc for doc in documents if doc.strip().startswith('PT ')]
        valid_doc_count = len(valid_documents)
        print(f"Precise detection of valid documents count: {valid_doc_count}")

        print("\nDocument structure verification (first 3):")
        for i, doc in enumerate(valid_documents[:3], 1):
            header = doc.strip().split('\n')[0]
            print(f"Document {i} start marker: {header[:20]}...")

        total_start = time.time()
        
        print(f"🚀 Using {max_workers} concurrent threads for processing")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tasks = [(i+1, doc) for i, doc in enumerate(valid_documents)]
            print(f"Actual documents to process: {len(tasks)}")
            results = executor.map(process_document, tasks)
            
        total_time = time.time() - total_start
        print(f"\nProcessing complete! Total documents: {valid_doc_count}")
        print(f"Total time: {total_time:.2f} seconds, Average per document: {total_time/valid_doc_count:.2f} seconds")

    error_file.close()