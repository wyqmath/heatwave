import csv
import json
import os
import time
import logging
import sys
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from openai import OpenAI
    import yaml
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install: pip install openai pyyaml tqdm")
    sys.exit(1)

def load_config():
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

LLM_API_KEY = os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE')
LLM_BASE_URL = ''
LLM_MODEL = ''

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_CSV = PROJECT_ROOT / "qa_all_hops_balanced.csv"
OUTPUT_RESULTS_CSV = PROJECT_ROOT / "ablation_no_kg_results.csv"
OUTPUT_METRICS_JSON = PROJECT_ROOT / "ablation_no_kg_metrics.json"

MAX_WORKERS = 4
MAX_RETRIES = 4000
API_TIMEOUT = 60
TARGET_RPM = 15

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ablation_no_kg.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class RPMController:
    
    def __init__(self, target_rpm: int = 30):
        import threading
        self.target_rpm = target_rpm
        self.min_interval = 60.0 / target_rpm
        self.request_times = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            current_rpm = len(self.request_times)
            
            wait_time = 0
            if current_rpm >= self.target_rpm:
                oldest_time = self.request_times[0]
                wait_time = 60 - (current_time - oldest_time) + 0.1
            elif self.request_times:
                last_time = self.request_times[-1]
                elapsed = current_time - last_time
                if elapsed < self.min_interval:
                    wait_time = self.min_interval - elapsed
            
            self.request_times.append(time.time() + wait_time)
        
        if wait_time > 0:
            time.sleep(wait_time)


rpm_controller = RPMController(TARGET_RPM)


class DirectLLMEvaluator:
    
    def __init__(self):
        self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    
    def answer_question(self, question: str, options: str) -> str:
        prompt = f"""Please answer the following medical multiple-choice question.

# Question:
{question}

# Options:
{options}

# Requirements:
1. Analyze based on your medical knowledge.
2. Select the most correct answer.
3. Output only the answer letter (A, B, C, or D), do not explain.

# Your Answer:"""

        for attempt in range(MAX_RETRIES + 1):
            try:
                rpm_controller.wait_if_needed()
                
                response = self.llm_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=10,
                    timeout=API_TIMEOUT
                )
                content = response.choices[0].message.content
                if content is None:
                    continue
                
                answer = content.strip().upper()
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in answer:
                        return letter
                        
            except Exception as e:
                logger.warning(f"LLM call failed (Attempt {attempt+1}/{MAX_RETRIES+1}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
        
        return "X"


def evaluate_single_question(args: Tuple[int, Dict, DirectLLMEvaluator]) -> Dict[str, Any]:
    idx, row, evaluator = args

    question = row.get('Question') or row.get('question', '')
    options = row.get('Options') or row.get('options', '')
    expected = (row.get('Answer') or row.get('answer', '')).strip().upper()
    hop = row.get('HopCount') or row.get('hop', '')

    predicted = evaluator.answer_question(question, options)
    is_correct = (predicted == expected)

    return {
        'index': idx,
        'question': question[:100] + '...' if len(question) > 100 else question,
        'expected': expected,
        'predicted': predicted,
        'correct': is_correct,
        'hop': str(hop) if hop else ''
    }


def calculate_metrics(results: List[Dict]) -> Dict[str, Any]:
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    failed = sum(1 for r in results if r['predicted'] == 'X')
    wrong = total - correct - failed

    valid_total = total - failed
    accuracy = correct / valid_total if valid_total > 0 else 0

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

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    prediction_distribution = Counter(r['predicted'] for r in results)
    expected_distribution = Counter(r['expected'] for r in results)

    hop_results = defaultdict(list)
    for r in results:
        hop = r.get('hop', 'unknown')
        if hop:
            hop_results[str(hop)].append(r)

    per_hop_metrics = {}
    for hop in sorted(hop_results.keys()):
        hop_data = hop_results[hop]
        hop_total = len(hop_data)
        hop_correct = sum(1 for r in hop_data if r['correct'])
        hop_failed = sum(1 for r in hop_data if r['predicted'] == 'X')
        hop_valid = hop_total - hop_failed
        hop_accuracy = hop_correct / hop_valid if hop_valid > 0 else 0
        per_hop_metrics[hop] = {
            'total': hop_total,
            'correct': hop_correct,
            'failed': hop_failed,
            'accuracy': round(hop_accuracy * 100, 2)
        }

    return {
        'total_questions': total,
        'correct_answers': correct,
        'wrong_answers': wrong,
        'failed_answers': failed,
        'accuracy': round(accuracy * 100, 2),
        'accuracy_raw': accuracy,
        'f1_score_macro': round(macro_f1, 3),
        'f1_per_class': {label: round(f1, 3) for label, f1 in zip(labels, f1_scores)},
        'prediction_distribution': dict(prediction_distribution),
        'expected_distribution': dict(expected_distribution),
        'per_hop_metrics': per_hop_metrics,
        'model': LLM_MODEL,
        'experiment_type': 'ablation_no_kg'
    }


def main():
    parser = argparse.ArgumentParser(description='Ablation study: Direct LLM reasoning without Knowledge Graph')
    parser.add_argument('--input', type=str, default=str(INPUT_CSV), help='Input CSV file path')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of evaluation questions (0 for all)')
    parser.add_argument('--start', type=int, default=0, help='Start from which question')
    parser.add_argument('--sample-per-hop', type=int, default=0, help='Number of samples per hop')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Ablation Study: Direct LLM Reasoning without Knowledge Graph (qwen3-max)")
    logger.info("=" * 60)

    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return

    logger.info(f"Loading test data: {input_file}")
    test_cases = []
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(row)

    if args.sample_per_hop > 0:
        hop_cases = defaultdict(list)
        for case in test_cases:
            hop = case.get('hop', '1')
            hop_cases[hop].append(case)

        sampled_cases = []
        import random
        for hop in sorted(hop_cases.keys()):
            cases = hop_cases[hop]
            if len(cases) > args.sample_per_hop:
                sampled_cases.extend(random.sample(cases, args.sample_per_hop))
            else:
                sampled_cases.extend(cases)
        test_cases = sampled_cases
        logger.info(f"Sampling by hop: {args.sample_per_hop} questions per hop, total {len(test_cases)} questions")

    if args.start > 0:
        test_cases = test_cases[args.start:]
        logger.info(f"Starting from question {args.start}")

    if args.limit > 0:
        test_cases = test_cases[:args.limit]
        logger.info(f"Limiting evaluation to {args.limit} questions")
    else:
        logger.info(f"Total {len(test_cases)} test data entries")

    logger.info("Initializing Direct LLM Evaluator (No KG)...")
    evaluator = DirectLLMEvaluator()

    logger.info(f"Starting evaluation using model: {LLM_MODEL}")
    logger.info(f"Concurrent threads: {MAX_WORKERS}, Target RPM: {TARGET_RPM}")
    results = []

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(evaluate_single_question, (i, row, evaluator)): i
                for i, row in enumerate(test_cases)
            }

            for future in tqdm(as_completed(futures), total=len(test_cases), desc="Evaluation Progress"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Error evaluating question {idx}: {e}")
                    results.append({
                        'index': idx, 'question': '(error)', 'expected': '',
                        'predicted': 'X', 'correct': False, 'hop': ''
                    })

        results.sort(key=lambda x: x['index'])

    except KeyboardInterrupt:
        logger.info("User interrupted evaluation")

    if not results:
        logger.error("No evaluation results")
        return

    metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("Ablation Study Results Summary (No KG)")
    print("=" * 60)
    print(f"Model: {metrics['model']}")
    print(f"Experiment Type: {metrics['experiment_type']}")
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Correct Answers: {metrics['correct_answers']}")
    print(f"Wrong Answers: {metrics['wrong_answers']}")
    print(f"API Failed: {metrics['failed_answers']}")
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"F1-Score (Macro): {metrics['f1_score_macro']}")
    print(f"F1 per class: {metrics['f1_per_class']}")
    print("-" * 60)
    print("Statistics by Hop:")
    for hop, hop_metrics in sorted(metrics.get('per_hop_metrics', {}).items()):
        print(f"  {hop}-hop: Total={hop_metrics['total']}, Correct={hop_metrics['correct']}, "
              f"Accuracy={hop_metrics['accuracy']}%")
    print("=" * 60)

    with open(OUTPUT_RESULTS_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    logger.info(f"Detailed results saved to: {OUTPUT_RESULTS_CSV}")

    with open(OUTPUT_METRICS_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Evaluation metrics saved to: {OUTPUT_METRICS_JSON}")


if __name__ == '__main__':
    main()