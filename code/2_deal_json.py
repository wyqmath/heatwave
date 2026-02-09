import os
import json
import re
import random

def normalize_key(key):
    key = re.sub(r'([a-z])([A-Z])', r'\1_\2', key)
    key = key.replace(' ', '_')
    key = re.sub(r'_+', '_', key)
    key = key.lower().strip('_')
    
    key_mapping = {
        'response_to': 'relationship',
        'relation': 'relationship',
        'response': 'relationship'
    }
    return key_mapping.get(key, key)

def clean_json_data(data):
    if isinstance(data, dict):
        return {normalize_key(k): clean_json_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_json_data(item) for item in data]
    return data

def process_json_files(folder_path):
    target_keys = {'start_node', 'relationship', 'end_node'}
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cleaned_data = clean_json_data(data)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
                
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")

def validate_json_structure(data):
    if isinstance(data, dict):
        keys = set(data.keys())
        required = {'start_node', 'relationship', 'end_node'}

        if not required.issubset(keys):
            missing = required - keys
            return False, f"Missing required fields: {missing}"

        for k in required:
            v = data[k]
            if not isinstance(v, str) or not v.strip():
                return False, f"Value of core field '{k}' must be a non-empty string"

        enhanced_fields = {
            'layer': str,
            'relation_type': str,
            'confidence': (int, float),
            'enhanced_timestamp': (int, float)
        }

        for field, expected_type in enhanced_fields.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    return False, f"Enhanced field '{field}' type error, expected: {expected_type.__name__ if isinstance(expected_type, type) else expected_type}"

        return True, ""
    
    if isinstance(data, list):
        for item in data:
            valid, msg = validate_json_structure(item)
            if not valid:
                return False, msg
        return True, ""
    return False, "Data structure should contain a dictionary or list"

def validate_json_files(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    
    total = len(all_files)
    success = 0
    failures = []
    
    print(f"\nStarting validation of {total} files...")
    
    for filename in all_files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            valid, msg = validate_json_structure(data)
            if valid:
                success += 1
            else:
                failures.append(f"{filename}: {msg}")
                
        except json.JSONDecodeError as e:
            failures.append(f"{filename} Line {e.lineno}: JSON format error - {e.msg}")
        except Exception as e:
            failures.append(f"{filename}: File parsing failed - {str(e)}")

    print(f"Validation completed! Success rate: {success/total:.1%}")
    print(f"Error details ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")

if __name__ == "__main__":
    import random
    
    enhanced_json_folder = './enhanced_json'
    original_json_folder = './json'

    if os.path.exists(enhanced_json_folder):
        json_folder = enhanced_json_folder
        print(f"Using enhanced JSON directory: {json_folder}")
    elif os.path.exists(original_json_folder):
        json_folder = original_json_folder
        print(f"Using original JSON directory: {json_folder}")
    else:
        print(f"Error: Neither {enhanced_json_folder} nor {original_json_folder} exists")
        exit(1)

    process_json_files(json_folder)
    validate_json_files(json_folder)