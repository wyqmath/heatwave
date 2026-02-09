import csv
import json
import time
import logging
import sys
import os
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import re
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from neo4j import GraphDatabase
    from openai import OpenAI
    import yaml
    import numpy as np
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def load_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

LLM_API_KEY = os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE')
LLM_BASE_URL = config.get('llm', {}).get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
LLM_MODEL = 'qwen3-max'

NEO4J_URI = config.get('knowledge_graph', {}).get('neo4j', {}).get('uri', 'bolt://localhost:7687')
NEO4J_USER = config.get('knowledge_graph', {}).get('neo4j', {}).get('user', 'neo4j')
NEO4J_PASSWORD = config.get('knowledge_graph', {}).get('neo4j', {}).get('password', os.environ.get('NEO4J_PASSWORD', ''))

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "qa_all_hops_balanced.csv"
NODES_CSV = PROJECT_ROOT / "all_nodes.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings.npy"
OUTPUT_RESULTS_CSV = PROJECT_ROOT / "kgqa_hybrid_results.csv"
OUTPUT_METRICS_JSON = PROJECT_ROOT / "evaluation_metrics.json" 

MAX_WORKERS = 2
TARGET_RPM = 15
API_TIMEOUT = 60

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[logging.FileHandler('kgqa_hybrid.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.ERROR)

class RPMController:
    def __init__(self, target_rpm: int = 15):
        import threading
        self.target_rpm = target_rpm
        self.min_interval = 60.0 / target_rpm
        self.request_times = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            if len(self.request_times) >= self.target_rpm:
                time.sleep(0.1)
            self.request_times.append(time.time())

rpm_controller = RPMController(TARGET_RPM)

class NodeRecommender:
    def __init__(self, csv_path: str, embed_path: str):
        import numpy as np
        import requests
        self.nodes = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.nodes.append(row['Node'])
        self.embeddings = np.load(embed_path)
        
        node_cfg = config.get('node_recommender', {})
        self.embedding_model = node_cfg.get('embedding_model', 'text-embedding-v4')
        self.embedding_api_key = node_cfg.get('embedding_api_key', LLM_API_KEY)
        self.embedding_api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        faiss_path = f"{embed_path}.faiss"
        if os.path.exists(faiss_path):
            try:
                import faiss
                self.index = faiss.read_index(faiss_path)
                self.use_faiss = True
                logger.info("FAISS index loaded")
            except:
                self.use_faiss = False
        else:
            self.use_faiss = False

    def _get_embedding(self, text: str) -> np.ndarray:
        import requests
        url = f"{self.embedding_api_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.embedding_api_key}", "Content-Type": "application/json"}
        data = {"model": self.embedding_model, "input": [text], "encoding_format": "float"}
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return np.array(response.json()['data'][0]['embedding'], dtype=np.float32)
        except:
            pass
        return np.zeros(1024, dtype=np.float32)

    def get_top_nodes(self, query: str, top_k: int = 5) -> List[Dict]:
        import faiss
        query_embedding = self._get_embedding(query)
        if self.use_faiss:
            query_vec = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vec)
            distances, indices = self.index.search(query_vec, top_k)
            return [{'Node': self.nodes[i], 'Score': float(distances[0][idx])} 
                    for idx, i in enumerate(indices[0]) if i < len(self.nodes)]
        return []

class KGQAHybridEvaluator:
    def __init__(self):
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            self.recommender = NodeRecommender(str(NODES_CSV), str(EMBEDDINGS_PATH))
        except:
            self.recommender = None

    def close(self):
        if self.neo4j_driver: self.neo4j_driver.close()

    def get_hybrid_context(self, question: str, options: str) -> str:
        if not self.recommender: return ""

        q_nodes = self.recommender.get_top_nodes(question, top_k=3)
        o_nodes = self.recommender.get_top_nodes(options, top_k=3)
        
        q_names = [n['Node'].replace("'", "\\'") for n in q_nodes]
        o_names = [n['Node'].replace("'", "\\'") for n in o_nodes]
        
        all_facts = set()
        
        with self.neo4j_driver.session() as session:
            if q_names:
                cypher = f"MATCH (n)-[r]-(m) WHERE n.name IN {json.dumps(q_names, ensure_ascii=False)} RETURN n.name, type(r), m.name LIMIT 8"
                try:
                    for rec in session.run(cypher):
                        all_facts.add(f"({rec['n.name']}) -[{rec['type(r)'].replace('_',' ')}]-> ({rec['m.name']})")
                except: pass
            
            if o_names:
                cypher = f"MATCH (n)-[r]-(m) WHERE n.name IN {json.dumps(o_names, ensure_ascii=False)} RETURN n.name, type(r), m.name LIMIT 8"
                try:
                    for rec in session.run(cypher):
                        all_facts.add(f"({rec['n.name']}) -[{rec['type(r)'].replace('_',' ')}]-> ({rec['m.name']})")
                except: pass

            if q_names and o_names:
                cypher = f"""
                MATCH (s), (e) WHERE s.name IN {json.dumps(q_names, ensure_ascii=False)} AND e.name IN {json.dumps(o_names, ensure_ascii=False)}
                MATCH p = shortestPath((s)-[*..4]-(e))
                RETURN nodes(p) as ns, relationships(p) as rs LIMIT 2
                """
                try:
                    for rec in session.run(cypher):
                        ns, rs = rec['ns'], rec['rs']
                        path = "".join([f"({ns[i]['name']}) -[{rs[i].type.replace('_',' ')}]-> " for i in range(len(rs))]) + f"({ns[-1]['name']})"
                        all_facts.add(f"PATH: {path}")
                except: pass

        if not all_facts: return ""
        sorted_facts = sorted(list(all_facts), key=lambda x: 0 if x.startswith("PATH") else 1)
        return "\n".join(sorted_facts[:20])

    def answer_question(self, question: str, options: str) -> str:
        context = self.get_hybrid_context(question, options)
        
        prompt = f"""Please answer the medical multiple-choice question based on the reference information.

# Reference:
{context}

# Question: {question}
# Options: {options}

# Requirements:
1. If there is direct evidence in the reference, please adopt it.
2. If the reference is invalid, please use internal knowledge.
3. Output the answer letter (A/B/C/D) directly, do not explain.

Answer:"""

        for _ in range(3):
            try:
                rpm_controller.wait_if_needed()
                res = self.llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=10, timeout=API_TIMEOUT
                )
                ans = res.choices[0].message.content.strip().upper()
                for c in ['A','B','C','D']:
                    if c in ans: return c
            except: time.sleep(1)
        return "X"

def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    failed = sum(1 for r in results if r['predicted'] == 'X')
    
    accuracy = correct / (total - failed) if (total - failed) > 0 else 0
    
    labels = ['A', 'B', 'C', 'D']
    f1_scores = []
    for label in labels:
        tp = sum(1 for r in results if r['predicted'] == label and r['expected'] == label)
        fp = sum(1 for r in results if r['predicted'] == label and r['expected'] != label)
        fn = sum(1 for r in results if r['expected'] == label and r['predicted'] != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores)
    
    hop_stats = {}
    by_hop = defaultdict(list)
    for r in results: by_hop[r['hop']].append(r)
    for h in sorted(by_hop.keys()):
        hc = sum(1 for r in by_hop[h] if r['correct'])
        ht = len(by_hop[h])
        hop_stats[h] = {
            'total': ht, 'correct': hc, 
            'accuracy': round(hc/ht*100, 2) if ht>0 else 0
        }

    return {
        'total_questions': total,
        'correct_answers': correct,
        'failed_answers': failed,
        'accuracy': round(accuracy * 100, 2),
        'f1_score_macro': round(macro_f1, 3),
        'f1_per_class': {l: round(f, 3) for l, f in zip(labels, f1_scores)},
        'prediction_distribution': dict(Counter(r['predicted'] for r in results)),
        'per_hop_metrics': hop_stats,
        'model': LLM_MODEL,
        'method': 'Hybrid (Neighbor+Path)'
    }

def evaluate_single_question(args):
    idx, row, evaluator = args
    q = row.get('Question') or row.get('question', '')
    opt = row.get('Options') or row.get('options', '')
    ans = (row.get('Answer') or row.get('answer', '')).strip().upper()
    hop = row.get('HopCount') or row.get('hop', '1')
    pred = evaluator.answer_question(q, opt)
    return {'index': idx, 'question': q, 'expected': ans, 'predicted': pred, 'correct': pred==ans, 'hop': str(hop)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(INPUT_CSV))
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    test_cases = []
    with open(args.input, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader: test_cases.append(row)
    
    if args.limit > 0: test_cases = test_cases[:args.limit]
    
    logger.info("="*60)
    logger.info(f"Starting KGQA Full Evaluation - Hybrid Strategy")
    logger.info(f"Total questions: {len(test_cases)}")
    logger.info("="*60)

    evaluator = KGQAHybridEvaluator()
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(evaluate_single_question, (i, r, evaluator)): i for i, r in enumerate(test_cases)}
        for future in tqdm(as_completed(futures), total=len(test_cases)):
            results.append(future.result())
            
    results.sort(key=lambda x: x['index'])
    metrics = calculate_metrics(results)
    
    print("\n" + "="*60)
    print("KGQA Final Evaluation Results (Hybrid Strategy)")
    print("="*60)
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"F1-Score (Macro): {metrics['f1_score_macro']}")
    print(f"Prediction Distribution: {metrics['prediction_distribution']}")
    print("-" * 60)
    for h, stats in metrics['per_hop_metrics'].items():
        print(f"  {h}-hop: {stats['correct']}/{stats['total']} = {stats['accuracy']}%")
    print("="*60)

    with open(OUTPUT_RESULTS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    with open(OUTPUT_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    evaluator.close()

if __name__ == '__main__':
    main()