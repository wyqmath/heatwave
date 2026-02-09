#!/usr/bin/env python3

import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripleEnhancer:
    
    def __init__(self, config_path: str = "config/default.yaml"):
        self.config = self._load_config(config_path)
        self.client = self._init_llm_client()
        
        self.layer_keywords = {
            "physical": [
                "temperature", "heat", "climate", "weather", "atmospheric", "soil", 
                "land", "surface", "radiation", "humidity", "precipitation", "wind",
                "heatwave", "warming", "cooling", "thermal", "evaporation", "moisture"
            ],
            "social": [
                "population", "health", "mortality", "morbidity", "human", "people",
                "community", "society", "demographic", "age", "elderly", "children",
                "adaptation", "behavior", "lifestyle", "vulnerability", "exposure"
            ],
            "economic": [
                "cost", "economic", "GDP", "industry", "agriculture", "energy",
                "electricity", "consumption", "production", "infrastructure", "urban",
                "development", "planning", "policy", "investment", "loss", "damage"
            ]
        }
        
        self.relation_types = {
            "causal": ["causes", "leads_to", "results_in", "triggers", "induces", "generates"],
            "influence": ["influences", "affects", "impacts", "modifies", "alters", "changes"],
            "correlation": ["correlates_with", "associated_with", "related_to", "linked_to"],
            "comparison": ["greater_than", "less_than", "similar_to", "different_from"],
            "temporal": ["precedes", "follows", "occurs_during", "coincides_with"],
            "spatial": ["located_in", "occurs_in", "distributed_in", "concentrated_in"]
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load configuration file {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "llm": {
                "api_key": os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE'),
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3-max"
            },
            "graph_enhancement": {
                "input_json_dir": "json",
                "output_enhanced_dir": "enhanced_json",
                "batch_size": 50,
                "max_workers": 10
            }
        }
    
    def _init_llm_client(self):
        llm_config = self.config.get("llm", {})
        return openai.OpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url")
        )
    
    def classify_layer_by_keywords(self, triple: Dict[str, str]) -> str:
        text = f"{triple['start_node']} {triple['relationship']} {triple['end_node']}".lower()
        
        layer_scores = {}
        for layer, keywords in self.layer_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            layer_scores[layer] = score
        
        if max(layer_scores.values()) == 0:
            return "unknown"
        
        return max(layer_scores, key=layer_scores.get)
    
    def classify_relation_type(self, relationship: str) -> str:
        relationship_lower = relationship.lower()
        
        for rel_type, relations in self.relation_types.items():
            if any(rel in relationship_lower for rel in relations):
                return rel_type
        
        return "other"
    
    def calculate_confidence_score(self, triple: Dict[str, str]) -> float:
        start_len = len(triple['start_node'].split())
        end_len = len(triple['end_node'].split())
        rel_len = len(triple['relationship'])
        
        entity_score = min(1.0, (start_len + end_len) / 6.0)
        
        relation_score = min(1.0, rel_len / 20.0)
        
        confidence = (entity_score * 0.6 + relation_score * 0.4)
        return round(confidence, 3)
    
    def llm_classify_layer(self, triples_batch: List[Dict[str, str]]) -> List[str]:
        try:
            triples_text = []
            for i, triple in enumerate(triples_batch):
                text = f"{i+1}. {triple['start_node']} -> {triple['relationship']} -> {triple['end_node']}"
                triples_text.append(text)
            
            prompt = f"""Please classify the following heatwave-related knowledge triples into the corresponding layers:
- physical: Physical environment layer (climate, temperature, atmosphere, soil, etc.)
- social: Social layer (population, health, community, adaptability, etc.)
- economic: Economic layer (cost, industry, infrastructure, policy, etc.)
- cross_layer: Cross-layer (involving relationships across multiple layers)

Triple list:
{chr(10).join(triples_text)}

Please return the classification results in the following JSON format:
{{"classifications": ["physical", "social", "economic", "cross_layer", ...]}}

Return only JSON, no other explanation."""

            response = self.client.chat.completions.create(
                model=self.config["llm"]["model"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            classifications = result.get("classifications", [])
            
            if len(classifications) != len(triples_batch):
                logger.warning(f"LLM classification result count mismatch, using keyword classification as fallback")
                return [self.classify_layer_by_keywords(triple) for triple in triples_batch]
            
            return classifications
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return [self.classify_layer_by_keywords(triple) for triple in triples_batch]
    
    def enhance_triples_batch(self, triples: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        layer_classifications = self.llm_classify_layer(triples)
        
        enhanced_triples = []
        for i, triple in enumerate(triples):
            enhanced = {
                **triple,
                "layer": layer_classifications[i] if i < len(layer_classifications) else "unknown",
                "relation_type": self.classify_relation_type(triple["relationship"]),
                "confidence": self.calculate_confidence_score(triple),
                "enhanced_timestamp": time.time()
            }
            enhanced_triples.append(enhanced)
        
        return enhanced_triples
    
    def enhance_json_file(self, input_file: Path, output_file: Path) -> bool:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                triples = json.load(f)
            
            if not triples:
                logger.warning(f"File {input_file} is empty")
                return False
            
            enhanced_triples = self.enhance_triples_batch(triples)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_triples, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process file {input_file}: {e}")
            return False
    
    def enhance_all_triples(self) -> Dict[str, Any]:
        config = self.config.get("graph_enhancement", {})
        input_dir = Path(config.get("input_json_dir", "json"))
        output_dir = Path(config.get("output_enhanced_dir", "enhanced_json"))
        max_workers = config.get("max_workers", 10)
        
        json_files = list(input_dir.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        if not json_files:
            logger.error("No JSON files found")
            return {"success": False, "message": "No JSON files found"}
        
        success_count = 0
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            for json_file in json_files:
                output_file = output_dir / json_file.name
                future = executor.submit(self.enhance_json_file, json_file, output_file)
                future_to_file[future] = json_file
            
            with tqdm(total=len(json_files), desc="Enhancing triples") as pbar:
                for future in as_completed(future_to_file):
                    json_file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_files.append(str(json_file))
                    except Exception as e:
                        logger.error(f"Exception occurred while processing file {json_file}: {e}")
                        failed_files.append(str(json_file))
                    
                    pbar.update(1)
                    time.sleep(0.1)
        
        stats = {
            "total_files": len(json_files),
            "success_count": success_count,
            "failed_count": len(failed_files),
            "success_rate": success_count / len(json_files) * 100,
            "failed_files": failed_files[:10],
            "output_directory": str(output_dir)
        }
        
        logger.info(f"Triple enhancement completed: {success_count}/{len(json_files)} successful")
        return {"success": True, "stats": stats}

def main():
    enhancer = TripleEnhancer()

    logger.info("Starting processing of all triple enhancements...")

    result = enhancer.enhance_all_triples()

    if result["success"]:
        stats = result["stats"]
        logger.info(f"✅ Triple enhancement completed!")
        logger.info(f"   Total files: {stats['total_files']}")
        logger.info(f"   Successfully processed: {stats['success_count']}")
        logger.info(f"   Failed count: {stats['failed_count']}")
        logger.info(f"   Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"   Output directory: {stats['output_directory']}")

        if stats['failed_files']:
            logger.warning(f"   Partial failed files: {stats['failed_files'][:5]}")
    else:
        logger.error(f"❌ Triple enhancement failed: {result.get('message', 'Unknown error')}")

    return result

if __name__ == "__main__":
    main()