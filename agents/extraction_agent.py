#!/usr/bin/env python3
"""
extraction_agent.py - HeDA抽取Agent
=====================================

负责论文七阶段协议中的前两个阶段：
  Stage 1: Corpus Acquisition & Filtering  (语料获取与过滤)
           → 调用 code/1_data_to_json.py  读取WOS论文，提取因果三元组
  Stage 2: Ontological Extraction          (本体抽取)
           → 调用 code/2_deal_json.py     规范化JSON结构，验证三元组完整性

工作流程：
  paper.txt → [LLM抽取因果三元组] → json/*.json → [结构验证/规范化] → 清洗后的json
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseAgent, ConfigManager, Stage, AgentState


class ExtractionAgent(BaseAgent):
    """抽取Agent - 负责从原始文献中提取结构化知识三元组
    
    对应论文架构中的 Extraction Agent，执行：
      1. 语料获取与过滤 (Stage 1): 从WOS导出的paper.txt中解析论文，
         利用LLM提取 (start_node, relationship, end_node) 因果三元组
      2. 本体抽取 (Stage 2): 规范化JSON键名，验证三元组结构完整性
    """

    def __init__(self):
        super().__init__(
            name="ExtractionAgent",
            stages=[Stage.CORPUS_ACQUISITION, Stage.ONTOLOGICAL_EXTRACTION]
        )
        # 从配置加载参数
        dp = self.config.get('data_processing', {})
        self.input_file = dp.get('input_file', 'paper.txt')
        self.output_dir = dp.get('output_dir', 'json/')
        self.max_workers = dp.get('max_workers', 15)

    # ----------------------------------------------------------
    #  Stage 1: 语料获取与过滤 + LLM因果三元组抽取
    # ----------------------------------------------------------

    def stage_1_corpus_acquisition(self) -> Dict[str, Any]:
        """Stage 1: 从paper.txt中读取论文，调用LLM提取因果三元组
        
        调用 code/1_data_to_json.py 的核心逻辑：
          - 读取WOS导出的paper.txt
          - 按 'ER  ' 分割为独立文档
          - 提取标题(TI)和摘要(AB)
          - 多线程调用LLM提取因果三元组
          - 保存为独立JSON文件
          
        Returns:
            包含 success, total_papers, extracted_count, output_dir 的结果字典
        """
        stage = Stage.CORPUS_ACQUISITION
        self._log_stage_start(stage)

        try:
            # 添加code目录到路径
            code_dir = Path(__file__).parent.parent / "code"
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))

            # 检查输入文件
            input_path = Path(self.input_file)
            if not input_path.exists():
                msg = f"输入文件不存在: {self.input_file}"
                self._log_stage_end(stage, False, msg)
                return {'success': False, 'message': msg}

            # 确保输出目录存在
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

            # 导入并调用 1_data_to_json.py 的核心函数
            from code import _data_to_json_module as dtj
            # 由于原始脚本是过程式的，我们直接复用其核心逻辑
            self.logger.info(f"读取输入文件: {self.input_file}")

            # 读取并分割论文
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = content.split('ER  ')
            documents = [doc.strip() for doc in documents if doc.strip()]
            total_papers = len(documents)
            self.logger.info(f"共解析到 {total_papers} 篇论文")

            # 提取标题和摘要
            papers = []
            for doc in documents:
                title, abstract = self._extract_ti_ab(doc)
                if title and abstract:
                    papers.append({'title': title, 'abstract': abstract})

            self.logger.info(f"有效论文(含标题+摘要): {len(papers)} 篇")

            # 多线程调用LLM提取三元组
            from concurrent.futures import ThreadPoolExecutor
            import threading

            lock = threading.Lock()
            extracted_count = 0
            failed_count = 0

            def process_paper(idx_paper):
                nonlocal extracted_count, failed_count
                idx, paper = idx_paper
                try:
                    text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
                    triplets = self._call_llm_extract(idx, text)
                    if triplets:
                        output_file = Path(self.output_dir) / f"{idx}.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(triplets, f, ensure_ascii=False, indent=2)
                        with lock:
                            extracted_count += 1
                    else:
                        with lock:
                            failed_count += 1
                except Exception as e:
                    self.logger.warning(f"论文 {idx}处理失败: {e}")
                    with lock:
                        failed_count += 1

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(process_paper, enumerate(papers))

            msg = f"成功提取 {extracted_count}/{len(papers)}篇，失败 {failed_count}篇"
            self._log_stage_end(stage, True, msg)
            return {
                'success': True,
                'total_papers': total_papers,
                'valid_papers': len(papers),
                'extracted_count': extracted_count,
                'failed_count': failed_count,
                'output_dir': self.output_dir,
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 1 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg}

    # ----------------------------------------------------------
    #  Stage 2: 本体抽取 - JSON结构验证与规范化
    # ----------------------------------------------------------

    def stage_2_ontological_extraction(self) -> Dict[str, Any]:
        """Stage 2: 规范化JSON结构，验证三元组完整性

        调用 code/2_deal_json.py 的核心逻辑：
          - 遍历json/目录下所有JSON文件
          - 规范化键名 (camelCase → snake_case)
          - 验证必需字段 (start_node, relationship, end_node)
          - 移除无效记录

        Returns:
            包含 success, total_files, valid_files, total_triplets 的结果字典
        """
        stage = Stage.ONTOLOGICAL_EXTRACTION
        self._log_stage_start(stage)

        try:
            json_dir = Path(self.output_dir)
            if not json_dir.exists():
                msg = f"JSON目录不存在: {self.output_dir}"
                self._log_stage_end(stage, False, msg)
                return {'success': False, 'message': msg}

            json_files = list(json_dir.glob("*.json"))
            if not json_files:
                msg = f"JSON目录为空: {self.output_dir}"
                self._log_stage_end(stage, False, msg)
                return {'success': False, 'message': msg}

            self.logger.info(f"发现 {len(json_files)} 个JSON文件待验证")

            # 键名映射表 (兼容各种命名风格)
            key_mapping = {
                'startNode': 'start_node', 'StartNode': 'start_node',
                'start': 'start_node', 'source': 'start_node',
                'endNode': 'end_node', 'EndNode': 'end_node',
                'end': 'end_node', 'target': 'end_node',
                'Relationship': 'relationship', 'relation': 'relationship',
                'edge': 'relationship', 'predicate': 'relationship',
                'Layer': 'layer', 'relationType': 'relation_type',
                'RelationType': 'relation_type', 'Confidence': 'confidence',
            }
            required_fields = {'start_node', 'relationship', 'end_node'}

            total_files = len(json_files)
            valid_files = 0
            total_triplets = 0
            invalid_files = 0

            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 确保是列表
                    if isinstance(data, dict):
                        data = [data]
                    if not isinstance(data, list):
                        invalid_files += 1
                        continue

                    # 规范化并验证
                    cleaned = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        # 键名规范化
                        normalized = {}
                        for k, v in item.items():
                            new_key = key_mapping.get(k, k)
                            normalized[new_key] = v
                        # 验证必需字段
                        if required_fields.issubset(normalized.keys()):
                            # 确保值非空
                            if all(str(normalized[f]).strip() for f in required_fields):
                                cleaned.append(normalized)

                    if cleaned:
                        with open(jf, 'w', encoding='utf-8') as f:
                            json.dump(cleaned, f, ensure_ascii=False, indent=2)
                        valid_files += 1
                        total_triplets += len(cleaned)
                    else:
                        invalid_files += 1

                except (json.JSONDecodeError, Exception) as e:
                    self.logger.warning(f"文件 {jf.name} 处理失败: {e}")
                    invalid_files += 1

            msg = (f"验证完成: {valid_files}/{total_files}个有效文件, "
                   f"共{total_triplets}条三元组, {invalid_files}个无效文件")
            self._log_stage_end(stage, True, msg)
            return {
                'success': True,
                'total_files': total_files,
                'valid_files': valid_files,
                'invalid_files': invalid_files,
                'total_triplets': total_triplets,
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 2 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg}

    # ----------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------

    def _extract_ti_ab(self, doc: str):
        """从WOS格式文档中提取标题(TI)和摘要(AB)"""
        title = ""
        abstract = ""
        lines = doc.split('\n')
        current_field = None

        for line in lines:
            if line.startswith('TI '):
                current_field = 'TI'
                title = line[3:].strip()
            elif line.startswith('AB '):
                current_field = 'AB'
                abstract = line[3:].strip()
            elif line.startswith('   ') and current_field:
                # 续行
                if current_field == 'TI':
                    title += ' ' + line.strip()
                elif current_field == 'AB':
                    abstract += ' ' + line.strip()
            elif line[:3].strip() and not line.startswith('   '):
                current_field = None

        return title.strip(), abstract.strip()

    def _call_llm_extract(self, idx: int, content: str) -> Optional[list]:
        """调用LLM提取因果三元组

        复用 code/1_data_to_json.py 中 call_with_messages 的核心逻辑
        """
        import re
        try:
            from openai import OpenAI
        except ImportError:
            self.logger.error("请安装 openai: pip install openai")
            return None

        llm_cfg = self.get_llm_config()
        client = OpenAI(api_key=llm_cfg['api_key'], base_url=llm_cfg['base_url'])

        prompt = f"""Please extract all causal relationships from the following academic text.
Return a JSON array where each element has exactly these fields:
- "start_node": the cause entity
- "relationship": the causal relationship description
- "end_node": the effect entity

Text:
{content}

Return ONLY a valid JSON array, no other text."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=llm_cfg['model'],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=llm_cfg['max_tokens'],
                    temperature=llm_cfg['temperature'],
                )
                result_text = response.choices[0].message.content.strip()

                # 清理markdown代码块
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0].strip()
                elif '```' in result_text:
                    result_text = result_text.split('```')[1].split('```')[0].strip()

                # 尝试提取JSON数组
                match = re.search(r'\[.*\]', result_text, re.DOTALL)
                if match:
                    result_text = match.group()

                triplets = json.loads(result_text)
                if isinstance(triplets, list) and len(triplets) > 0:
                    self.logger.debug(f"论文 {idx}: 提取到 {len(triplets)}条三元组")
                    return triplets

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.logger.warning(f"论文 {idx}LLM调用失败: {e}")

        return None

    # ----------------------------------------------------------
    #  主执行入口
    # ----------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """执行抽取Agent的完整流程: Stage 1 → Stage 2

        Returns:
            包含两个阶段执行结果的汇总字典
        """
        self.state = AgentState.RUNNING
        self.logger.info(f"{'#'*60}")
        self.logger.info(f"# ExtractionAgent 启动")
        self.logger.info(f"# 负责阶段: Stage 1 (语料获取) → Stage 2 (本体抽取)")
        self.logger.info(f"{'#'*60}")

        results = {}

        # Stage 1: 语料获取与过滤
        r1 = self.stage_1_corpus_acquisition()
        results['stage_1'] = r1
        if not r1.get('success'):
            self.state = AgentState.FAILED
            self.logger.error("Stage 1 失败，终止执行")
            return {'success': False, 'results': results}

        # Stage 2: 本体抽取
        r2 = self.stage_2_ontological_extraction()
        results['stage_2'] = r2

        # 汇总
        overall_success = r1.get('success') and r2.get('success')
        self.state = AgentState.SUCCESS if overall_success else AgentState.FAILED

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"ExtractionAgent 执行完毕 - {'成功 ✔' if overall_success else '失败 ✘'}")
        self.logger.info(f"{'='*60}")

        return {
            'success': overall_success,
            'agent': self.name,
            'results': results,
        }


# ============================================================
#  独立运行入口
# ============================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    agent = ExtractionAgent()
    result = agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

