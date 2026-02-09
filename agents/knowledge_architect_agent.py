#!/usr/bin/env python3
"""
knowledge_architect_agent.py - HeDA知识架构Agent
==================================================

负责论文七阶段协议中的中间三个阶段：
  Stage 3: Semantic Disambiguation   (语义消歧)
           → 调用 code/4_com_nodes.py       收集所有唯一节点
           → 调用 code/5_nodes_group.py      基于语义相似度分组
           → 调用 code/6_clean_and_map_nodes.py  LLM生成标准化名称
  Stage 4: Attribute Enrichment      (属性增强)
           → 调用 code/3_graph_enhancement.py 添加层分类/关系类型/置信度
           → 调用 code/7_standardized_json.py 应用标准化节点名称
  Stage 5: Topological Construction  (拓扑构建)
           → 调用 code/8_upload_neo4j.py     上传知识图谱到Neo4j

工作流程：
  json/ → [节点收集] → [语义分组] → [标准化命名] → [属性增强] → [名称替换]
       → enhanced_json/ → [Neo4j上传] → 图数据库
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


class KnowledgeArchitectAgent(BaseAgent):
    """知识架构Agent - 负责知识图谱的语义消歧、属性增强和拓扑构建

    对应论文架构中的 Knowledge Architect Agent，执行：
      1. 语义消歧 (Stage 3): 收集节点→语义分组→LLM标准化命名
      2. 属性增强 (Stage 4): 层分类(physical/social/economic)、关系类型、置信度
      3. 拓扑构建 (Stage 5): 上传到Neo4j图数据库
    """

    def __init__(self):
        super().__init__(
            name="KnowledgeArchitectAgent",
            stages=[
                Stage.SEMANTIC_DISAMBIGUATION,
                Stage.ATTRIBUTE_ENRICHMENT,
                Stage.TOPOLOGICAL_CONSTRUCTION,
            ]
        )
        paths = self.config.get('paths', {})
        self.json_dir = paths.get('json_dir', 'json/')
        self.enhanced_dir = paths.get('enhanced_json_dir', 'enhanced_json/')
        self.nodes_file = paths.get('nodes_file', 'all_nodes.csv')
        self.mapping_file = paths.get('mapping_file', 'mapping.csv')

    # ----------------------------------------------------------
    #  Stage 3: 语义消歧
    # ----------------------------------------------------------

    def stage_3_semantic_disambiguation(self) -> Dict[str, Any]:
        """Stage 3: 语义消歧 - 收集节点、语义分组、标准化命名

        子步骤：
          3a. 收集所有唯一节点 (code/4_com_nodes.py)
          3b. 基于语义相似度分组 (code/5_nodes_group.py)
          3c. LLM生成标准化名称 (code/6_clean_and_map_nodes.py)
        """
        stage = Stage.SEMANTIC_DISAMBIGUATION
        self._log_stage_start(stage)

        try:
            # --- 3a: 收集所有唯一节点 ---
            self.logger.info("  [3a] 收集所有唯一节点...")
            nodes_result = self._collect_all_nodes()
            if not nodes_result['success']:
                self._log_stage_end(stage, False, nodes_result['message'])
                return nodes_result
            self.logger.info(f"  [3a] 收集到 {nodes_result['node_count']} 个唯一节点")

            # --- 3b: 语义相似度分组 ---
            self.logger.info("  [3b] 基于语义相似度进行节点分组...")
            group_result = self._semantic_grouping()
            if not group_result['success']:
                self._log_stage_end(stage, False, group_result['message'])
                return group_result
            self.logger.info(f"  [3b] 生成 {group_result['group_count']} 个语义分组")

            # --- 3c: LLM标准化命名 ---
            self.logger.info("  [3c] 调用LLM生成标准化节点名称...")
            naming_result = self._standardize_names()
            if not naming_result['success']:
                self._log_stage_end(stage, False, naming_result['message'])
                return naming_result
            self.logger.info(f"  [3c] 生成 {naming_result['mapping_count']} 条名称映射")

            msg = (f"语义消歧完成: {nodes_result['node_count']}个节点 → "
                   f"{group_result['group_count']}个分组 → "
                   f"{naming_result['mapping_count']}条映射")
            self._log_stage_end(stage, True, msg)
            return {
                'success': True,
                'node_count': nodes_result['node_count'],
                'group_count': group_result['group_count'],
                'mapping_count': naming_result['mapping_count'],
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 3 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg}

    # ----------------------------------------------------------
    #  Stage 4: 属性增强
    # ----------------------------------------------------------

    def stage_4_attribute_enrichment(self) -> Dict[str, Any]:
        """Stage 4: 属性增强 - 添加层分类、关系类型、置信度，并应用标准化名称

        子步骤：
          4a. 三元组增强 (code/3_graph_enhancement.py):
              为每条三元组添加 layer(physical/social/economic)、
              relation_type、confidence 属性
          4b. 应用标准化名称 (code/7_standardized_json.py):
              用mapping.csv中的标准名称替换原始节点名
        """
        stage = Stage.ATTRIBUTE_ENRICHMENT
        self._log_stage_start(stage)

        try:
            # --- 4a: 三元组属性增强 ---
            self.logger.info("  [4a] 为三元组添加层分类/关系类型/置信度...")
            enhance_result = self._enhance_triplets()
            if not enhance_result['success']:
                self._log_stage_end(stage, False, enhance_result['message'])
                return enhance_result
            self.logger.info(f"  [4a] 增强完成: {enhance_result['enhanced_count']}条三元组")

            # --- 4b: 应用标准化名称 ---
            self.logger.info("  [4b] 应用标准化节点名称...")
            std_result = self._apply_standardized_names()
            if not std_result['success']:
                self._log_stage_end(stage, False, std_result['message'])
                return std_result
            self.logger.info(f"  [4b] 名称替换完成: {std_result['replaced_count']}个节点被替换")

            msg = (f"属性增强完成: {enhance_result['enhanced_count']}条三元组增强, "
                   f"{std_result['replaced_count']}个节点名称标准化")
            self._log_stage_end(stage, True, msg)
            return {
                'success': True,
                'enhanced_count': enhance_result['enhanced_count'],
                'replaced_count': std_result['replaced_count'],
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 4 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg}

    # ----------------------------------------------------------
    #  Stage 5: 拓扑构建
    # ----------------------------------------------------------

    def stage_5_topological_construction(self) -> Dict[str, Any]:
        """Stage 5: 拓扑构建 - 上传知识图谱到Neo4j图数据库

        调用 code/8_upload_neo4j.py 的核心逻辑：
          - 读取enhanced_json/目录下所有JSON三元组
          - 按关系类型分组
          - 批量MERGE导入Neo4j
          - 创建索引
        """
        stage = Stage.TOPOLOGICAL_CONSTRUCTION
        self._log_stage_start(stage)

        try:
            neo4j_cfg = self.get_neo4j_config()
            self.logger.info(f"  连接Neo4j: {neo4j_cfg['uri']}")

            # 收集所有三元组
            source_dir = Path(self.enhanced_dir)
            if not source_dir.exists():
                source_dir = Path(self.json_dir)

            json_files = list(source_dir.glob("*.json"))
            if not json_files:
                msg = f"未找到JSON文件: {source_dir}"
                self._log_stage_end(stage, False, msg)
                return {'success': False, 'message': msg}

            all_triplets = []
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_triplets.extend(data)
                    elif isinstance(data, dict):
                        all_triplets.append(data)
                except Exception:
                    continue

            self.logger.info(f"  共加载 {len(all_triplets)}条三元组，准备上传Neo4j...")

            # 导入Neo4j驱动
            try:
                from neo4j import GraphDatabase
            except ImportError:
                msg = "请安装neo4j驱动: pip install neo4j"
                self._log_stage_end(stage, False, msg)
                return {'success': False, 'message': msg}

            # 连接并上传
            import re
            driver = GraphDatabase.driver(
                neo4j_cfg['uri'],
                auth=(neo4j_cfg['user'], neo4j_cfg['password'])
            )

            node_count = 0
            edge_count = 0
            batch_size = self.config.get('neo4j', {}).get('batch_size', 2000)

            with driver.session() as session:
                # 清空数据库（可选）
                self.logger.info("  清空现有数据...")
                session.run("MATCH (n) DETACH DELETE n")

                # 批量导入
                for i in range(0, len(all_triplets), batch_size):
                    batch = all_triplets[i:i+batch_size]
                    for triplet in batch:
                        start = triplet.get('start_node', '').strip()
                        end = triplet.get('end_node', '').strip()
                        rel = triplet.get('relationship', 'RELATES_TO').strip()
                        layer = triplet.get('layer', 'unknown')
                        confidence = triplet.get('confidence', 0.5)

                        if not start or not end:
                            continue

                        # 清理关系名称（Neo4j要求合法标识符）
                        rel_clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel).upper()
                        if not rel_clean or rel_clean[0].isdigit():
                            rel_clean = 'REL_' + rel_clean

                        query = f"""
                        MERGE (a:Entity {{name: $start}})
                        MERGE (b:Entity {{name: $end}})
                        MERGE (a)-[r:`{rel_clean}`]->(b)
                        SET r.description = $rel_desc,
                            r.layer = $layer,
                            r.confidence = $confidence
                        """
                        session.run(query, start=start, end=end,
                                   rel_desc=rel, layer=layer, confidence=confidence)
                        edge_count += 1

                    self.logger.info(f"  已上传 {min(i+batch_size, len(all_triplets))}/{len(all_triplets)}")

                # 创建索引
                session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")

                # 统计节点数
                result = session.run("MATCH (n) RETURN count(n) as cnt")
                node_count = result.single()['cnt']

            driver.close()

            msg = f"拓扑构建完成: {node_count}个节点, {edge_count}条边已上传Neo4j"
            self._log_stage_end(stage, True, msg)
            return {
                'success': True,
                'node_count': node_count,
                'edge_count': edge_count,
                'message': msg,
            }

        except Exception as e:
            msg = f"Stage 5 执行异常: {str(e)}"
            self._log_stage_end(stage, False, msg)
            return {'success': False, 'message': msg}

    # ==============================================================
    #  辅助方法 - Stage 3 子步骤
    # ==============================================================

    def _collect_all_nodes(self) -> Dict[str, Any]:
        """3a: 收集所有唯一节点 (对应 code/4_com_nodes.py)

        遍历json/目录，提取所有start_node和end_node，去重后输出CSV
        """
        try:
            json_dir = Path(self.json_dir)
            json_files = list(json_dir.glob("*.json"))
            if not json_files:
                return {'success': False, 'message': f"JSON目录为空: {self.json_dir}"}

            all_nodes = set()
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
                    for item in data:
                        if isinstance(item, dict):
                            sn = item.get('start_node', '').strip()
                            en = item.get('end_node', '').strip()
                            if sn:
                                all_nodes.add(sn)
                            if en:
                                all_nodes.add(en)
                except Exception:
                    continue

            # 输出到CSV
            output_file = Path('combination_nodes.csv')
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['node'])
                for node in sorted(all_nodes):
                    writer.writerow([node])

            return {
                'success': True,
                'node_count': len(all_nodes),
                'output_file': str(output_file),
            }

        except Exception as e:
            return {'success': False, 'message': f"节点收集失败: {e}"}

    def _semantic_grouping(self) -> Dict[str, Any]:
        """3b: 基于语义相似度分组 (对应 code/5_nodes_group.py)

        使用SentenceTransformer + FAISS进行语义相似度分组：
          - 加载combination_nodes.csv中的节点
          - 用all-MiniLM-L6-v2编码为向量
          - 构建FAISS IVF索引
          - 按余弦相似度阈值分组
          - 输出output_relation_nodes.csv
        """
        try:
            nodes_file = Path('combination_nodes.csv')
            if not nodes_file.exists():
                return {'success': False, 'message': "combination_nodes.csv不存在，请先执行3a"}

            # 读取节点
            nodes = []
            with open(nodes_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    node = row.get('node', '').strip()
                    if node:
                        nodes.append(node)

            if len(nodes) < 2:
                return {'success': False, 'message': f"节点数量不足: {len(nodes)}"}

            self.logger.info(f"    加载 {len(nodes)}个节点，开始语义编码...")

            # 语义编码
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                import faiss
            except ImportError as ie:
                return {'success': False, 'message': f"缺少依赖: {ie}. 请安装 sentence-transformers faiss-cpu"}

            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(nodes, show_progress_bar=True, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype='float32')

            # 构建FAISS索引
            dim = embeddings.shape[1]
            n_cells = min(100, len(nodes) // 10 + 1)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_cells, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings)
            index.add(embeddings)
            index.nprobe = min(20, n_cells)

            # 相似度搜索与分组
            similarity_threshold = 0.85
            k = min(50, len(nodes))
            distances, indices = index.search(embeddings, k)

            # Union-Find 分组
            parent = list(range(len(nodes)))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x, y):
                px, py = find(x), find(y)
                if px != py:
                    parent[px] = py

            for i in range(len(nodes)):
                for j_idx in range(k):
                    j = indices[i][j_idx]
                    if j < 0 or j >= len(nodes):
                        continue
                    if distances[i][j_idx] >= similarity_threshold:
                        union(i, j)

            # 构建分组
            groups = defaultdict(list)
            for i in range(len(nodes)):
                groups[find(i)].append(nodes[i])

            # 输出到CSV
            output_file = Path('output_relation_nodes.csv')
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['group_id', 'nodes'])
                for gid, (_, members) in enumerate(sorted(groups.items(), key=lambda x: -len(x[1]))):
                    writer.writerow([gid, '|'.join(members)])

            multi_groups = sum(1 for g in groups.values() if len(g) > 1)
            return {
                'success': True,
                'group_count': len(groups),
                'multi_node_groups': multi_groups,
                'output_file': str(output_file),
            }

        except Exception as e:
            return {'success': False, 'message': f"语义分组失败: {e}"}


    def _standardize_names(self) -> Dict[str, Any]:
        """3c: LLM生成标准化名称 (对应 code/6_clean_and_map_nodes.py)"""
        try:
            group_file = Path('output_relation_nodes.csv')
            if not group_file.exists():
                return {'success': False, 'message': "output_relation_nodes.csv不存在"}

            groups = []
            with open(group_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    members = row.get('nodes', '').split('|')
                    members = [m.strip() for m in members if m.strip()]
                    if len(members) > 1:
                        groups.append(members)

            if not groups:
                mapping_file = Path(self.mapping_file)
                with open(mapping_file, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['original', 'standardized'])
                return {'success': True, 'mapping_count': 0}

            self.logger.info(f"    需要标准化的分组: {len(groups)}个")

            from openai import OpenAI
            llm_cfg = self.get_llm_config()
            client = OpenAI(api_key=llm_cfg['api_key'], base_url=llm_cfg['base_url'])

            mapping = {}
            for i, group in enumerate(groups):
                try:
                    prompt = (
                        f"以下是一组语义相似的实体名称，请生成一个最佳的标准化英文名称。\n"
                        f"只返回标准化名称，不要其他内容。\n\n"
                        f"实体列表: {', '.join(group)}"
                    )
                    response = client.chat.completions.create(
                        model=llm_cfg['model'],
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=100, temperature=0.3,
                    )
                    std_name = response.choices[0].message.content.strip().strip('"\'')
                    if std_name:
                        for original in group:
                            mapping[original] = std_name
                except Exception as e:
                    self.logger.warning(f"    分组 {i}标准化失败: {e}")
                    std_name = min(group, key=len)
                    for original in group:
                        mapping[original] = std_name

            mapping_file = Path(self.mapping_file)
            with open(mapping_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['original', 'standardized'])
                for orig, std in sorted(mapping.items()):
                    writer.writerow([orig, std])

            return {'success': True, 'mapping_count': len(mapping),
                    'output_file': str(mapping_file)}
        except Exception as e:
            return {'success': False, 'message': f"标准化命名失败: {e}"}

    # ==============================================================
    #  辅助方法 - Stage 4 子步骤
    # ==============================================================

    def _enhance_triplets(self) -> Dict[str, Any]:
        """4a: 三元组属性增强 (对应 code/3_graph_enhancement.py)

        为每条三元组添加 layer / relation_type / confidence，输出到 enhanced_json/
        """
        try:
            json_dir = Path(self.json_dir)
            enhanced_dir = Path(self.enhanced_dir)
            enhanced_dir.mkdir(parents=True, exist_ok=True)

            json_files = list(json_dir.glob("*.json"))
            if not json_files:
                return {'success': False, 'message': f"JSON目录为空: {self.json_dir}"}

            layer_keywords = {
                'physical': ['temperature', 'heat', 'drought', 'rainfall', 'precipitation',
                            'flood', 'storm', 'wind', 'humidity', 'radiation', 'climate',
                            'weather', 'thermal', 'warming', 'cooling', 'ocean', 'sea'],
                'social': ['health', 'mortality', 'morbidity', 'population', 'community',
                          'urban', 'rural', 'migration', 'displacement', 'vulnerability',
                          'adaptation', 'resilience', 'equity', 'access', 'education'],
                'economic': ['gdp', 'income', 'cost', 'loss', 'productivity', 'labor',
                            'agriculture', 'crop', 'yield', 'market', 'trade', 'price',
                            'infrastructure', 'energy', 'supply', 'demand', 'investment'],
            }

            enhanced_count = 0
            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
                    if not isinstance(data, list):
                        continue

                    enhanced_data = []
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        text = f"{item.get('start_node','')} {item.get('relationship','')}{item.get('end_node','')}".lower()
                        layer = 'unknown'
                        max_score = 0
                        for l, keywords in layer_keywords.items():
                            score = sum(1 for kw in keywords if kw in text)
                            if score > max_score:
                                max_score = score
                                layer = l

                        item['layer'] = layer
                        item['relation_type'] = 'causal'
                        item['confidence'] = min(0.5 + max_score * 0.1, 1.0)
                        item['enhanced_timestamp'] = int(__import__('time').time())
                        enhanced_data.append(item)
                        enhanced_count += 1

                    output_file = enhanced_dir / jf.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(enhanced_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    self.logger.warning(f"    增强文件 {jf.name}失败: {e}")

            return {'success': True, 'enhanced_count': enhanced_count,
                    'output_dir': str(enhanced_dir)}
        except Exception as e:
            return {'success': False, 'message': f"三元组增强失败: {e}"}

    def _apply_standardized_names(self) -> Dict[str, Any]:
        """4b: 应用标准化名称 (对应 code/7_standardized_json.py)

        读取mapping.csv，遍历enhanced_json/中的文件，替换节点名称
        """
        try:
            mapping_file = Path(self.mapping_file)
            if not mapping_file.exists():
                return {'success': True, 'replaced_count': 0, 'message': "无映射文件，跳过名称替换"}

            # 加载映射
            name_map = {}
            with open(mapping_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    orig = row.get('original', '').strip()
                    std = row.get('standardized', '').strip()
                    if orig and std:
                        name_map[orig] = std

            if not name_map:
                return {'success': True, 'replaced_count': 0, 'message': "映射表为空"}

            enhanced_dir = Path(self.enhanced_dir)
            if not enhanced_dir.exists():
                enhanced_dir = Path(self.json_dir)

            json_files = list(enhanced_dir.glob("*.json"))
            replaced_count = 0

            for jf in json_files:
                try:
                    with open(jf, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]

                    modified = False
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        for field in ['start_node', 'end_node']:
                            old_val = item.get(field, '')
                            if old_val in name_map:
                                item[field] = name_map[old_val]
                                replaced_count += 1
                                modified = True

                    if modified:
                        with open(jf, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                except Exception:
                    continue

            return {'success': True, 'replaced_count': replaced_count}
        except Exception as e:
            return {'success': False, 'message': f"名称替换失败: {e}"}

    # ==============================================================
    #  主执行入口
    # ==============================================================

    def run(self) -> Dict[str, Any]:
        """执行知识架构Agent的完整流程: Stage 3 → Stage 4 → Stage 5"""
        self.state = AgentState.RUNNING
        self.logger.info(f"{'#'*60}")
        self.logger.info(f"# KnowledgeArchitectAgent 启动")
        self.logger.info(f"# 负责阶段: Stage 3 (语义消歧) → Stage 4 (属性增强) → Stage 5 (拓扑构建)")
        self.logger.info(f"{'#'*60}")

        results = {}

        # Stage 3
        r3 = self.stage_3_semantic_disambiguation()
        results['stage_3'] = r3
        if not r3.get('success'):
            self.state = AgentState.FAILED
            self.logger.error("Stage 3 失败，终止执行")
            return {'success': False, 'results': results}

        # Stage 4
        r4 = self.stage_4_attribute_enrichment()
        results['stage_4'] = r4
        if not r4.get('success'):
            self.state = AgentState.FAILED
            self.logger.error("Stage 4 失败，终止执行")
            return {'success': False, 'results': results}

        # Stage 5
        r5 = self.stage_5_topological_construction()
        results['stage_5'] = r5

        overall = r3.get('success') and r4.get('success') and r5.get('success')
        self.state = AgentState.SUCCESS if overall else AgentState.FAILED

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"KnowledgeArchitectAgent 执行完毕 - {'成功 ✔' if overall else '失败 ✘'}")
        self.logger.info(f"{'='*60}")

        return {'success': overall, 'agent': self.name, 'results': results}


# ============================================================
#  独立运行入口
# ============================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    agent = KnowledgeArchitectAgent()
    result = agent.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))