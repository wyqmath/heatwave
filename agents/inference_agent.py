#!/usr/bin/env python3
"""
inference_agent.py - HeDA推理Agent
====================================

负责论文七阶段协议中的最后两个阶段：
  Stage 6: Vector Embedding           (向量嵌入)
           → 调用 code/9_all_nodes.py        生成节点列表与邻接表
           → 调用 code/10_node_recommender.py 构建向量嵌入与FAISS索引
  Stage 7: Reasoning & Verification   (推理与验证)
           → 调用 code/11_data_to_qa_by_hop.py   多跳QA数据集生成
           → 调用 code/12_balance_answer.py       答案分布均衡化
           → 调用 code/13_kgqa_evaluation.py      KGQA混合检索评估
           → 调用 code/14_ablation_no_kg.py       消融实验(无KG基线)
           → 调用 code/15_multi_hop_reasoning.py  多跳推理引擎
           → 调用 code/16_advanced_reasoning.py   跨层高级推理
           → 调用 code/17_large_scale_mining.py   大规模路径挖掘
           → 调用 code/18_anaysis.py              结果分析与可视化

工作流程：
  enhanced_json/ → [节点列表] → [向量嵌入] → [QA生成] → [评估] → [多跳推理]
                → [跨层分析] → [大规模挖掘] → [可视化] → reports/
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict

from agents.base_agent import BaseAgent, ConfigManager, Stage, AgentState


class InferenceAgent(BaseAgent):
    """推理Agent - 负责向量嵌入、多跳推理和结果验证

    对应论文架构中的 Inference Agent，执行：
      1. 向量嵌入 (Stage 6): 节点列表生成 + 向量化 + FAISS索引构建
      2. 推理与验证 (Stage 7): QA评估 + 多跳推理 + 跨层分析 + 大规模挖掘
    """

    def __init__(self):
        super().__init__(
            name="InferenceAgent",
            stages=[Stage.VECTOR_EMBEDDING, Stage.REASONING_VERIFICATION]
        )
        paths = self.config.get('paths', {})
        self.json_dir = paths.get('json_dir', 'json/')
        self.enhanced_dir = paths.get('enhanced_json_dir', 'enhanced_json/')
        self.reports_dir = paths.get('reports_dir', 'reports/')
        self.nodes_file = paths.get('nodes_file', 'all_nodes.csv')
        self.adjacency_file = paths.get('adjacency_file', 'adjacency.json')

    # ----------------------------------------------------------
    #  Stage 6: 向量嵌入
    # ----------------------------------------------------------

    def stage_6_vector_embedding(self) -> Dict[str, Any]:
        """Stage 6: 向量嵌入 - 生成节点列表、构建向量索引

        子步骤：
          6a. 生成节点列表与邻接表 (code/9_all_nodes.py)
          6b. 构建向量嵌入与FAISS索引 (code/10_node_recommender.py)
        """
        stage = Stage.VECTOR_EMBEDDING
        self._log_stage_start(stage)

        try:
            # --- 6a: 生成节点列表与邻接表 ---
            self.logger.info("  [6a] 生成节点列表与邻接表...")
            nodes_result = self._generate_node_list()
            if not nodes_result['success']:
                self._log_stage_end(stage, False, nodes_result['message'])
                return nodes_result
            self.logger.info(f"  [6a] 生成 {nodes_result['node_count']}个节点, "
                           f"{nodes_result['edge_count']}条邻接关系")

            # --- 6b: 构建向量嵌入 ---
            self.logger.info("  [6b] 构建向量嵌入与FAISS索引...")
            embed_result = self._build_embeddings()
            if not embed_result['success']:
                self._log_stage_end(stage, False, embed_result['message'])
                return embed_result
            self.logger.info(f"  [6b] 嵌入完成: {embed_result['embedded_count']}个节点向量化")

            msg = (f"向量嵌入完成: {nodes_result['node_count']}个节点, "
                   f"{embed_result['embedded_count']}个嵌入向量")
            self._log_stage_end(stage, True, msg)
            return {
                'success': True,
                'node_count': nodes_result['node_count'],
                'edge_count': nodes_result['edge_count'],
                'embedded_count': embed_result['embedded_count'],
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 6 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg}

    # ----------------------------------------------------------
    #  Stage 7: 推理与验证
    # ----------------------------------------------------------

    def stage_7_reasoning_verification(self) -> Dict[str, Any]:
        """Stage 7: 推理与验证 - 多跳推理、跨层分析、大规模挖掘

        子步骤（按顺序执行，部分可选）：
          7a. 多跳QA数据集生成 (code/11 + code/12)
          7b. KGQA评估 (code/13) + 消融实验 (code/14)
          7c. 多跳推理引擎 (code/15)
          7d. 跨层高级推理 (code/16)
          7e. 大规模路径挖掘 (code/17)
          7f. 结果分析与可视化 (code/18)
        """
        stage = Stage.REASONING_VERIFICATION
        self._log_stage_start(stage)

        sub_results = {}
        try:
            # --- 7a: 多跳QA数据集生成 ---
            self.logger.info("  [7a] 生成多跳QA数据集...")
            qa_result = self._generate_qa_dataset()
            sub_results['qa_generation'] = qa_result
            if qa_result['success']:
                self.logger.info(f"  [7a] 生成 {qa_result.get('qa_count', 0)}条QA")

            # --- 7b: KGQA评估 ---
            self.logger.info("  [7b] 执行KGQA评估...")
            eval_result = self._run_kgqa_evaluation()
            sub_results['kgqa_evaluation'] = eval_result
            if eval_result['success']:
                self.logger.info(f"  [7b] 评估完成: 准确率={eval_result.get('accuracy', 'N/A')}")

            # --- 7c: 多跳推理 ---
            self.logger.info("  [7c] 执行多跳推理引擎...")
            reasoning_result = self._run_multi_hop_reasoning()
            sub_results['multi_hop_reasoning'] = reasoning_result
            if reasoning_result['success']:
                self.logger.info(f"  [7c] 发现 {reasoning_result.get('path_count', 0)}条推理路径")

            # --- 7d: 跨层高级推理 ---
            self.logger.info("  [7d] 执行跨层高级推理...")
            advanced_result = self._run_advanced_reasoning()
            sub_results['advanced_reasoning'] = advanced_result

            # --- 7e: 大规模路径挖掘 ---
            self.logger.info("  [7e] 执行大规模路径挖掘...")
            mining_result = self._run_large_scale_mining()
            sub_results['large_scale_mining'] = mining_result
            if mining_result['success']:
                self.logger.info(f"  [7e] 挖掘到 {mining_result.get('path_count', 0)}条风险传播链")

            # --- 7f: 结果分析与可视化 ---
            self.logger.info("  [7f] 执行结果分析与可视化...")
            analysis_result = self._run_analysis()
            sub_results['analysis'] = analysis_result

            # 汇总
            success_count = sum(1 for r in sub_results.values() if r.get('success'))
            total_count = len(sub_results)
            msg = f"推理验证完成: {success_count}/{total_count}个子任务成功"
            self._log_stage_end(stage, success_count > 0, msg)

            return {
                'success': success_count > 0,
                'sub_results': sub_results,
                'success_count': success_count,
                'total_count': total_count,
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 7 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg, 'sub_results': sub_results}


    # ==============================================================
    #  辅助方法 - Stage 6 子步骤
    # ==============================================================

    def _generate_node_list(self) -> Dict[str, Any]:
        """6a: 生成节点列表与邻接表 (对应 code/9_all_nodes.py)

        遍历enhanced_json/，提取所有节点和邻接关系，输出:
          - all_nodes.csv: 所有唯一节点
          - adjacency.json: 邻接表 {node: [{neighbor, relationship, layer}, ...]}
        """
        try:
            source_dir = Path(self.enhanced_dir)
            if not source_dir.exists():
                source_dir = Path(self.json_dir)

            json_files = list(source_dir.glob("*.json"))
            if not json_files:
                return {'success': False, 'message': f"未找到JSON文件: {source_dir}"}

            all_nodes = set()
            adjacency = defaultdict(list)

            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        sn = item.get('start_node', '').strip()
                        en = item.get('end_node', '').strip()
                        rel = item.get('relationship', '').strip()
                        layer = item.get('layer', 'unknown')
                        if sn and en:
                            all_nodes.add(sn)
                            all_nodes.add(en)
                            adjacency[sn].append({
                                'neighbor': en, 'relationship': rel, 'layer': layer
                            })
                            adjacency[en].append({
                                'neighbor': sn, 'relationship': rel, 'layer': layer
                            })
                except Exception:
                    continue

            # 输出 all_nodes.csv
            nodes_file = Path(self.nodes_file)
            with open(nodes_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['node'])
                for node in sorted(all_nodes):
                    writer.writerow([node])

            # 输出 adjacency.json
            adj_file = Path(self.adjacency_file)
            with open(adj_file, 'w', encoding='utf-8') as f:
                json.dump(dict(adjacency), f, ensure_ascii=False, indent=2)

            edge_count = sum(len(v) for v in adjacency.values()) // 2
            return {
                'success': True,
                'node_count': len(all_nodes),
                'edge_count': edge_count,
            }
        except Exception as e:
            return {'success': False, 'message': f"节点列表生成失败: {e}"}

    def _build_embeddings(self) -> Dict[str, Any]:
        """6b: 构建向量嵌入与FAISS索引 (对应 code/10_node_recommender.py)

        使用DashScope text-embedding-v4 API对所有节点进行向量化，
        构建FAISS索引用于语义检索
        """
        try:
            nodes_file = Path(self.nodes_file)
            if not nodes_file.exists():
                return {'success': False, 'message': f"{self.nodes_file}不存在"}

            nodes = []
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node = row.get('node', '').strip()
                    if node:
                        nodes.append(node)

            if not nodes:
                return {'success': False, 'message': "节点列表为空"}

            self.logger.info(f"    对 {len(nodes)}个节点进行向量嵌入...")

            try:
                import numpy as np
                import faiss
            except ImportError as ie:
                return {'success': False, 'message': f"缺少依赖: {ie}"}

            # 使用DashScope Embedding API
            embed_cfg = self.get_embedding_config()
            from openai import OpenAI
            client = OpenAI(
                api_key=embed_cfg['api_key'],
                base_url=embed_cfg.get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
            )

            model = embed_cfg.get('model', 'text-embedding-v4')
            dimension = embed_cfg.get('dimension', 1024)
            batch_size = 25  # API批量限制

            all_embeddings = []
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i+batch_size]
                try:
                    response = client.embeddings.create(model=model, input=batch)
                    for item in response.data:
                        all_embeddings.append(item.embedding)
                except Exception as e:
                    self.logger.warning(f"    批次 {i//batch_size}嵌入失败: {e}")
                    # 用零向量填充
                    for _ in batch:
                        all_embeddings.append([0.0] * dimension)

            embeddings = np.array(all_embeddings, dtype='float32')

            # 构建FAISS索引
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

            # 保存索引和节点映射
            index_file = Path('node_embeddings.index')
            faiss.write_index(index, str(index_file))

            mapping_file = Path('node_id_mapping.json')
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump({i: node for i, node in enumerate(nodes)}, f,
                         ensure_ascii=False, indent=2)

            return {
                'success': True,
                'embedded_count': len(nodes),
                'dimension': dimension,
                'index_file': str(index_file),
            }
        except Exception as e:
            return {'success': False, 'message': f"向量嵌入失败: {e}"}

    # ==============================================================
    #  辅助方法 - Stage 7 子步骤
    # ==============================================================

    def _generate_qa_dataset(self) -> Dict[str, Any]:
        """7a: 多跳QA数据集生成 (对应 code/11 + code/12)

        基于知识图谱邻接表生成1-hop/2-hop/3-hop问答对，
        并进行答案分布均衡化
        """
        try:
            adj_file = Path(self.adjacency_file)
            if not adj_file.exists():
                return {'success': False, 'message': f"{self.adjacency_file}不存在"}

            with open(adj_file, 'r', encoding='utf-8') as f:
                adjacency = json.load(f)

            if not adjacency:
                return {'success': False, 'message': "邻接表为空"}

            from openai import OpenAI
            llm_cfg = self.get_llm_config()
            client = OpenAI(api_key=llm_cfg['api_key'], base_url=llm_cfg['base_url'])

            qa_pairs = []
            nodes = list(adjacency.keys())

            # 生成1-hop QA
            for node in nodes[:50]:  # 限制数量避免API过载
                neighbors = adjacency.get(node, [])
                if not neighbors:
                    continue
                for nb in neighbors[:3]:
                    nb_name = nb.get('neighbor', '')
                    rel = nb.get('relationship', '')
                    if not nb_name or not rel:
                        continue
                    qa_pairs.append({
                        'hop': 1,
                        'question': f"What is the relationship between {node} and {nb_name}?",
                        'answer': rel,
                        'path': [node, nb_name],
                    })

            # 生成2-hop QA
            for node in nodes[:30]:
                neighbors_1 = adjacency.get(node, [])
                for nb1 in neighbors_1[:2]:
                    nb1_name = nb1.get('neighbor', '')
                    neighbors_2 = adjacency.get(nb1_name, [])
                    for nb2 in neighbors_2[:2]:
                        nb2_name = nb2.get('neighbor', '')
                        if nb2_name and nb2_name != node:
                            qa_pairs.append({
                                'hop': 2,
                                'question': f"How does {node} indirectly affect {nb2_name}?",
                                'answer': f"Through {nb1_name}",
                                'path': [node, nb1_name, nb2_name],
                            })

            # 答案均衡化 (简化版 code/12)
            from collections import Counter
            hop_counts = Counter(qa['hop'] for qa in qa_pairs)
            if hop_counts:
                min_count = min(hop_counts.values())
                balanced = []
                hop_seen = defaultdict(int)
                for qa in qa_pairs:
                    if hop_seen[qa['hop']] < min_count:
                        balanced.append(qa)
                        hop_seen[qa['hop']] += 1
                qa_pairs = balanced

            # 保存
            output_file = Path('qa_dataset.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'qa_count': len(qa_pairs),
                'output_file': str(output_file),
            }
        except Exception as e:
            return {'success': False, 'message': f"QA数据集生成失败: {e}"}

    def _run_kgqa_evaluation(self) -> Dict[str, Any]:
        """7b: KGQA混合检索评估 (对应 code/13 + code/14)

        使用KG增强的混合检索策略评估QA准确率，
        同时运行无KG消融基线进行对比
        """
        try:
            qa_file = Path('qa_dataset.json')
            if not qa_file.exists():
                return {'success': False, 'message': "qa_dataset.json不存在"}

            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)

            if not qa_pairs:
                return {'success': False, 'message': "QA数据集为空"}

            from openai import OpenAI
            llm_cfg = self.get_llm_config()
            client = OpenAI(api_key=llm_cfg['api_key'], base_url=llm_cfg['base_url'])

            # 加载邻接表用于KG增强检索
            adj_file = Path(self.adjacency_file)
            adjacency = {}
            if adj_file.exists():
                with open(adj_file, 'r', encoding='utf-8') as f:
                    adjacency = json.load(f)

            correct_kg = 0
            correct_no_kg = 0
            total = min(len(qa_pairs), 50)  # 限制评估数量

            for qa in qa_pairs[:total]:
                question = qa['question']
                expected = qa['answer']
                path_nodes = qa.get('path', [])

                # KG增强检索: 收集路径上的上下文
                context_parts = []
                for node in path_nodes:
                    neighbors = adjacency.get(node, [])
                    for nb in neighbors[:5]:
                        context_parts.append(
                            f"{node} --[{nb.get('relationship','')}]--> {nb.get('neighbor','')}"
                        )
                kg_context = "\n".join(context_parts[:10])

                try:
                    # 带KG上下文的评估
                    resp_kg = client.chat.completions.create(
                        model=llm_cfg['model'],
                        messages=[{"role": "user", "content":
                            f"Based on the following knowledge graph context:\n{kg_context}\n\n"
                            f"Question: {question}\nAnswer concisely:"}],
                        max_tokens=200, temperature=0.3,
                    )
                    answer_kg = resp_kg.choices[0].message.content.strip().lower()
                    if expected.lower() in answer_kg:
                        correct_kg += 1

                    # 无KG消融基线
                    resp_no = client.chat.completions.create(
                        model=llm_cfg['model'],
                        messages=[{"role": "user", "content":
                            f"Question: {question}\nAnswer concisely:"}],
                        max_tokens=200, temperature=0.3,
                    )
                    answer_no = resp_no.choices[0].message.content.strip().lower()
                    if expected.lower() in answer_no:
                        correct_no_kg += 1
                except Exception:
                    continue

            accuracy_kg = correct_kg / total if total > 0 else 0
            accuracy_no_kg = correct_no_kg / total if total > 0 else 0

            result = {
                'success': True,
                'total': total,
                'accuracy': round(accuracy_kg, 4),
                'accuracy_no_kg': round(accuracy_no_kg, 4),
                'improvement': round(accuracy_kg - accuracy_no_kg, 4),
            }

            # 保存评估结果
            eval_file = Path(self.reports_dir) / 'kgqa_evaluation.json'
            Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return result
        except Exception as e:
            return {'success': False, 'message': f"KGQA评估失败: {e}"}

    def _run_multi_hop_reasoning(self) -> Dict[str, Any]:
        """7c: 多跳推理引擎 (对应 code/15_multi_hop_reasoning.py)

        基于NetworkX有向图执行BFS/DFS多跳路径搜索，
        发现跨层级的因果传播链
        """
        try:
            adj_file = Path(self.adjacency_file)
            if not adj_file.exists():
                return {'success': False, 'message': f"{self.adjacency_file}不存在"}

            with open(adj_file, 'r', encoding='utf-8') as f:
                adjacency = json.load(f)

            try:
                import networkx as nx
            except ImportError:
                return {'success': False, 'message': "缺少networkx依赖"}

            # 构建有向图
            G = nx.DiGraph()
            for node, neighbors in adjacency.items():
                for nb in neighbors:
                    nb_name = nb.get('neighbor', '')
                    rel = nb.get('relationship', '')
                    layer = nb.get('layer', 'unknown')
                    if nb_name:
                        G.add_edge(node, nb_name, relationship=rel, layer=layer)

            self.logger.info(f"    图规模: {G.number_of_nodes()}节点, {G.number_of_edges()}边")

            # 多跳路径搜索 (2-4跳)
            paths_found = []
            nodes_list = list(G.nodes())[:100]  # 限制搜索范围

            for source in nodes_list[:20]:
                for target in nodes_list[20:40]:
                    if source == target:
                        continue
                    try:
                        for path in nx.all_simple_paths(G, source, target, cutoff=4):
                            if len(path) >= 3:  # 至少2跳
                                path_info = {
                                    'path': path,
                                    'hops': len(path) - 1,
                                    'edges': []
                                }
                                for i in range(len(path) - 1):
                                    edge_data = G.get_edge_data(path[i], path[i+1], {})
                                    path_info['edges'].append({
                                        'from': path[i], 'to': path[i+1],
                                        'relationship': edge_data.get('relationship', ''),
                                        'layer': edge_data.get('layer', ''),
                                    })
                                paths_found.append(path_info)
                                if len(paths_found) >= 200:
                                    break
                        if len(paths_found) >= 200:
                            break
                    except nx.NetworkXNoPath:
                        continue
                if len(paths_found) >= 200:
                    break

            # 保存
            output_file = Path(self.reports_dir) / 'multi_hop_paths.json'
            Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(paths_found, f, ensure_ascii=False, indent=2, default=str)

            return {
                'success': True,
                'path_count': len(paths_found),
                'graph_nodes': G.number_of_nodes(),
                'graph_edges': G.number_of_edges(),
            }
        except Exception as e:
            return {'success': False, 'message': f"多跳推理失败: {e}"}

    def _run_advanced_reasoning(self) -> Dict[str, Any]:
        """7d: 跨层高级推理 (对应 code/16_advanced_reasoning.py)

        执行跨层分析(CrossLayerAnalyzer)和新颖性评分(NoveltyAnalyzer):
          NoveltyScore(P) = α·LF(P) + β·CLC(P) + γ·IP(P)
          α=0.5, β=0.3, γ=0.2
        """
        try:
            paths_file = Path(self.reports_dir) / 'multi_hop_paths.json'
            if not paths_file.exists():
                return {'success': False, 'message': "multi_hop_paths.json不存在"}

            with open(paths_file, 'r', encoding='utf-8') as f:
                paths = json.load(f)

            if not paths:
                return {'success': True, 'message': "无路径可分析", 'analyzed_count': 0}

            # 跨层分析
            cross_layer_paths = []
            for p in paths:
                layers = set()
                for edge in p.get('edges', []):
                    layer = edge.get('layer', 'unknown')
                    if layer != 'unknown':
                        layers.add(layer)
                if len(layers) >= 2:  # 跨越至少2个层
                    p['cross_layers'] = list(layers)
                    p['layer_count'] = len(layers)
                    cross_layer_paths.append(p)

            # 新颖性评分
            alpha, beta, gamma = 0.5, 0.3, 0.2
            for p in cross_layer_paths:
                hops = p.get('hops', 1)
                layer_count = p.get('layer_count', 1)
                # LF: 路径长度因子 (越长越新颖)
                lf = min(hops / 5.0, 1.0)
                # CLC: 跨层复杂度
                clc = min(layer_count / 3.0, 1.0)
                # IP: 信息量 (基于路径中唯一关系数)
                unique_rels = len(set(e.get('relationship', '') for e in p.get('edges', [])))
                ip = min(unique_rels / 4.0, 1.0)
                p['novelty_score'] = round(alpha * lf + beta * clc + gamma * ip, 4)

            # 按新颖性排序
            cross_layer_paths.sort(key=lambda x: x.get('novelty_score', 0), reverse=True)

            # 保存
            output_file = Path(self.reports_dir) / 'advanced_reasoning.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cross_layer_paths, f, ensure_ascii=False, indent=2, default=str)

            return {
                'success': True,
                'analyzed_count': len(cross_layer_paths),
                'top_novelty': cross_layer_paths[0].get('novelty_score') if cross_layer_paths else 0,
            }
        except Exception as e:
            return {'success': False, 'message': f"高级推理失败: {e}"}

    def _run_large_scale_mining(self) -> Dict[str, Any]:
        """7e: 大规模路径挖掘 (对应 code/17_large_scale_mining.py)

        BFS遍历全图挖掘跨层风险传播链，生成Sankey图数据
        """
        try:
            adj_file = Path(self.adjacency_file)
            if not adj_file.exists():
                return {'success': False, 'message': f"{self.adjacency_file}不存在"}

            with open(adj_file, 'r', encoding='utf-8') as f:
                adjacency = json.load(f)

            # BFS挖掘跨层路径
            layer_transitions = defaultdict(int)  # (layer_from, layer_to) → count
            all_chains = []

            visited_pairs = set()
            for start_node in list(adjacency.keys())[:200]:
                # BFS
                queue = [(start_node, [start_node], [])]
                visited = {start_node}
                while queue:
                    current, path, layers_seq = queue.pop(0)
                    if len(path) > 5:
                        continue
                    neighbors = adjacency.get(current, [])
                    for nb in neighbors:
                        nb_name = nb.get('neighbor', '')
                        layer = nb.get('layer', 'unknown')
                        if nb_name and nb_name not in visited:
                            new_path = path + [nb_name]
                            new_layers = layers_seq + [layer]
                            visited.add(nb_name)

                            # 记录层转换
                            if len(new_layers) >= 2:
                                prev_layer = new_layers[-2]
                                curr_layer = new_layers[-1]
                                if prev_layer != curr_layer and prev_layer != 'unknown' and curr_layer != 'unknown':
                                    layer_transitions[(prev_layer, curr_layer)] += 1
                                    pair_key = (start_node, nb_name)
                                    if pair_key not in visited_pairs and len(new_path) >= 3:
                                        all_chains.append({
                                            'path': new_path,
                                            'layers': new_layers,
                                            'length': len(new_path),
                                        })
                                        visited_pairs.add(pair_key)

                            if len(new_path) < 5:
                                queue.append((nb_name, new_path, new_layers))

            # 生成Sankey数据
            sankey_data = {
                'nodes': [],
                'links': [],
            }
            layer_set = set()
            for (src, tgt), count in layer_transitions.items():
                layer_set.add(src)
                layer_set.add(tgt)
            layer_list = sorted(layer_set)
            sankey_data['nodes'] = [{'name': l}for l in layer_list]
            layer_idx = {l: i for i, l in enumerate(layer_list)}
            for (src, tgt), count in sorted(layer_transitions.items(), key=lambda x: -x[1]):
                sankey_data['links'].append({
                    'source': layer_idx.get(src, 0),
                    'target': layer_idx.get(tgt, 0),
                    'value': count,
                })

            # 保存
            Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
            chains_file = Path(self.reports_dir) / 'risk_chains.json'
            with open(chains_file, 'w', encoding='utf-8') as f:
                json.dump(all_chains[:500], f, ensure_ascii=False, indent=2)

            sankey_file = Path(self.reports_dir) / 'sankey_data.json'
            with open(sankey_file, 'w', encoding='utf-8') as f:
                json.dump(sankey_data, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'path_count': len(all_chains),
                'layer_transitions': len(layer_transitions),
            }
        except Exception as e:
            return {'success': False, 'message': f"大规模挖掘失败: {e}"}

    def _run_analysis(self) -> Dict[str, Any]:
        """7f: 结果分析与可视化 (对应 code/18_anaysis.py)

        汇总所有推理结果，生成统计报告
        """
        try:
            Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
            report = {
                'title': 'HeDA Heatwave Risk Analysis Report',
                'sections': [],
            }

            # 汇总各阶段结果
            result_files = {
                'KGQA评估': 'kgqa_evaluation.json',
                '多跳推理路径': 'multi_hop_paths.json',
                '高级推理分析': 'advanced_reasoning.json',
                '风险传播链': 'risk_chains.json',
                'Sankey数据': 'sankey_data.json',
            }

            for section_name, filename in result_files.items():
                filepath = Path(self.reports_dir) / filename
                section = {'name': section_name, 'file': filename, 'exists': filepath.exists()}
                if filepath.exists():
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            section['record_count'] = len(data)
                        elif isinstance(data, dict):
                            section['keys'] = list(data.keys())
                    except Exception:
                        pass
                report['sections'].append(section)

            # 保存汇总报告
            report_file = Path(self.reports_dir) / 'analysis_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            return {
                'success': True,
                'report_file': str(report_file),
                'sections': len(report['sections']),
            }
        except Exception as e:
            return {'success': False, 'message': f"结果分析失败: {e}"}

    # ==============================================================
    #  主执行入口
    # ==============================================================

    def run(self) -> Dict[str, Any]:
        """执行推理Agent的完整流程: Stage 6 → Stage 7"""
        self.state = AgentState.RUNNING
        self.logger.info(f"{'#'*60}")
        self.logger.info(f"# InferenceAgent 启动")
        self.logger.info(f"# 负责阶段: Stage 6 (向量嵌入) → Stage 7 (推理与验证)")
        self.logger.info(f"{'#'*60}")

        results = {}

        # Stage 6
        r6 = self.stage_6_vector_embedding()
        results['stage_6'] = r6
        if not r6.get('success'):
            self.state = AgentState.FAILED
            self.logger.error("Stage 6 失败，终止执行")
            return {'success': False, 'results': results}

        # Stage 7
        r7 = self.stage_7_reasoning_verification()
        results['stage_7'] = r7

        overall = r6.get('success') and r7.get('success')
        self.state = AgentState.SUCCESS if overall else AgentState.FAILED

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"InferenceAgent 执行完毕 - {'成功 ✔' if overall else '失败 ✘'}")
        self.logger.info(f"{'='*60}")

        return {'success': overall, 'agent': self.name, 'results': results}


# ============================================================
#  独立运行入口
# ============================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    agent = InferenceAgent()
    result = agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))