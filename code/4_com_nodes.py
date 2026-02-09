#!/usr/bin/env python3

import json
import os
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_nodes():
    unique_nodes = set()
    
    enhanced_json_dir = Path('./enhanced_json')
    original_json_dir = Path('./json')

    if enhanced_json_dir.exists() and list(enhanced_json_dir.glob('*.json')):
        json_dir = enhanced_json_dir
        logger.info(f"✅ Using enhanced JSON directory: {json_dir}")
    elif original_json_dir.exists() and list(original_json_dir.glob('*.json')):
        json_dir = original_json_dir
        logger.info(f"⚠️  Using original JSON directory: {json_dir}")
    else:
        logger.error(f"❌ Error: Neither {enhanced_json_dir} nor {original_json_dir} exists")
        return
    
    json_files = list(json_dir.glob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files")
    
    total_relations = 0
    valid_relations = 0
    invalid_relations = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for relation in data:
                    total_relations += 1
                    
                    normalized_fields = {
                        key.lower().replace(" ", "").replace("_", ""): value 
                        for key, value in relation.items()
                    }
                    
                    start = normalized_fields.get('startnode')
                    end = normalized_fields.get('endnode')
                    
                    if start and end:
                        if isinstance(start, str) and isinstance(end, str):
                            unique_nodes.add(start.strip())
                            unique_nodes.add(end.strip())
                            valid_relations += 1
                        else:
                            logger.warning(f"Non-string nodes found in file {json_file.name}: start={type(start)}, end={type(end)}")
                            invalid_relations += 1
                    else:
                        logger.warning(f"Invalid relation (missing fields) in file {json_file.name}: {relation}")
                        invalid_relations += 1
                        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed for file {json_file.name}: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing file {json_file.name}: {str(e)}")
    
    unique_nodes = {node for node in unique_nodes if node and node.strip()}
    
    nodes_df = pd.DataFrame(sorted(list(unique_nodes)), columns=['Node'])
    
    output_file = 'combination_nodes.csv'
    nodes_df.to_csv(output_file, index=False, encoding='utf-8')
    
    logger.info("=" * 60)
    logger.info("✅ Node collection completed!")
    logger.info(f"   Files processed: {len(json_files)}")
    logger.info(f"   Total relations: {total_relations}")
    logger.info(f"   Valid relations: {valid_relations}")
    logger.info(f"   Invalid relations: {invalid_relations}")
    logger.info(f"   Unique nodes: {len(unique_nodes)}")
    logger.info(f"   Output file: {output_file}")
    logger.info("=" * 60)
    
    return len(unique_nodes)

def main():
    logger.info("🚀 Starting node collection...")
    logger.info("=" * 60)
    
    try:
        node_count = collect_nodes()
        if node_count and node_count > 0:
            logger.info("🎉 Node collection script executed successfully!")
            return 0
        else:
            logger.error("❌ No nodes collected")
            return 1
    except Exception as e:
        logger.error(f"❌ Script execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())