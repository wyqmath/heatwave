#!/usr/bin/env python3

import csv
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def rotate_options(options: str, original_answer: str, target_answer: str) -> Tuple[str, str]:
    if original_answer.upper() == target_answer.upper():
        return options, target_answer.upper()
    
    option_parts = []
    for opt in options.split(' | '):
        opt = opt.strip()
        if opt and len(opt) >= 3 and opt[1] == '.':
            content = opt[2:].strip()
            option_parts.append(content)
    
    if len(option_parts) != 4:
        logger.warning(f"Unable to parse options: {options[:50]}...")
        return options, original_answer
    
    answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    orig_idx = answer_map.get(original_answer.upper(), 0)
    target_idx = answer_map.get(target_answer.upper(), 0)
    
    option_parts[orig_idx], option_parts[target_idx] = option_parts[target_idx], option_parts[orig_idx]
    
    letters = ['A', 'B', 'C', 'D']
    new_options = ' | '.join([f"{letters[i]}. {option_parts[i]}" for i in range(4)])
    
    return new_options, target_answer.upper()


def balance_dataset(input_file: Path, output_file: Path, target_per_hop: int = 1000) -> Dict[str, Any]:
    target_per_answer = target_per_hop // 4
    
    logger.info(f"Reading input file: {input_file}")
    data_by_hop = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hop = row.get('HopCount') or row.get('hop', '1')
            data_by_hop[str(hop)].append(row)
    
    logger.info("Original data distribution:")
    original_stats = {}
    for hop in sorted(data_by_hop.keys()):
        answer_counts = defaultdict(int)
        for row in data_by_hop[hop]:
            ans = (row.get('Answer') or row.get('answer', '')).strip().upper()
            answer_counts[ans] += 1
        original_stats[hop] = dict(answer_counts)
        logger.info(f"  {hop}-hop: Total={len(data_by_hop[hop])}, Distribution={dict(sorted(answer_counts.items()))}")
    
    balanced_data = []
    balanced_stats = {}
    target_answers = ['A', 'B', 'C', 'D']
    
    for hop in sorted(data_by_hop.keys()):
        hop_data = data_by_hop[hop].copy()
        random.shuffle(hop_data)
        
        if len(hop_data) < target_per_hop:
            logger.warning(f"Insufficient data for {hop}-hop: {len(hop_data)} < {target_per_hop}")
            while len(hop_data) < target_per_hop:
                hop_data.extend(random.choices(data_by_hop[hop], k=min(100, target_per_hop - len(hop_data))))
        
        hop_balanced = []
        for i in range(target_per_hop):
            row = hop_data[i].copy()
            target_answer = target_answers[i % 4]
            
            original_answer = (row.get('Answer') or row.get('answer', '')).strip().upper()
            options = row.get('Options') or row.get('options', '')
            
            new_options, new_answer = rotate_options(options, original_answer, target_answer)
            
            if 'Options' in row:
                row['Options'] = new_options
                row['Answer'] = new_answer
            else:
                row['options'] = new_options
                row['answer'] = new_answer
            
            hop_balanced.append(row)
        
        balanced_data.extend(hop_balanced)
        
        answer_counts = defaultdict(int)
        for row in hop_balanced:
            ans = (row.get('Answer') or row.get('answer', '')).strip().upper()
            answer_counts[ans] += 1
        balanced_stats[hop] = dict(answer_counts)
        logger.info(f"  {hop}-hop balanced: Total={len(hop_balanced)}, Distribution={dict(sorted(answer_counts.items()))}")

    logger.info(f"\nWriting balanced data: {output_file}")
    if balanced_data:
        fieldnames = list(balanced_data[0].keys())
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(balanced_data)
        logger.info(f"✅ Saved {len(balanced_data)} balanced data entries")

    return {
        'original_stats': original_stats,
        'balanced_stats': balanced_stats,
        'total_original': sum(len(v) for v in data_by_hop.values()),
        'total_balanced': len(balanced_data)
    }


def main():
    parser = argparse.ArgumentParser(description='Balance QA dataset answer distribution')
    parser.add_argument('--input', type=str, default=str(PROJECT_ROOT / 'qa_all_hops.csv'),
                        help='Input CSV file path')
    parser.add_argument('--output', type=str, default=str(PROJECT_ROOT / 'qa_all_hops_balanced.csv'),
                        help='Output CSV file path')
    parser.add_argument('--target-per-hop', type=int, default=1000,
                        help='Target number of questions per hop (default 1000, 250 each for ABCD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default 42, ensures reproducibility)')
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("=" * 60)
    logger.info("QA Dataset Answer Distribution Balancing Tool")
    logger.info("=" * 60)

    input_file = Path(args.input)
    output_file = Path(args.output)

    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return 1

    stats = balance_dataset(input_file, output_file, args.target_per_hop)

    print("\n" + "=" * 60)
    print("Balance Results Summary")
    print("=" * 60)
    print(f"Total original data: {stats['total_original']}")
    print(f"Total balanced data: {stats['total_balanced']}")
    print("\nAnswer distribution by hop:")
    for hop in sorted(stats['balanced_stats'].keys()):
        print(f"  {hop}-hop: {stats['balanced_stats'][hop]}")
    print("=" * 60)
    print(f"\n✅ Balanced data saved to: {output_file}")

    return 0


if __name__ == '__main__':
    exit(main())