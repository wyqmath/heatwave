#!/usr/bin/env python3
"""
HeDA (Heatwave Disaster Analysis) - 多Agent协同系统
=====================================================

基于论文架构的四Agent层级系统：
  - CoordinatorAgent: 中央协调单元，管理动态任务调度
  - ExtractionAgent:  抽取Agent，负责语料获取/过滤(Stage1) + 本体抽取(Stage2)
  - KnowledgeArchitectAgent: 知识架构Agent，负责语义消歧(Stage3) + 属性增强(Stage4) + 拓扑构建(Stage5)
  - InferenceAgent:   推理Agent，负责向量嵌入(Stage6) + 推理验证(Stage7)

七阶段协议 (Seven-Stage Protocol):
  Stage 1: Corpus Acquisition & Filtering  (语料获取与过滤)
  Stage 2: Ontological Extraction          (本体抽取)
  Stage 3: Semantic Disambiguation         (语义消歧)
  Stage 4: Attribute Enrichment            (属性增强)
  Stage 5: Topological Construction        (拓扑构建)
  Stage 6: Vector Embedding                (向量嵌入)
  Stage 7: Reasoning & Verification        (推理与验证)
"""

from agents.base_agent import BaseAgent, ConfigManager
from agents.extraction_agent import ExtractionAgent
from agents.knowledge_architect_agent import KnowledgeArchitectAgent
from agents.inference_agent import InferenceAgent
from agents.coordinator_agent import CoordinatorAgent

__all__ = [
    'BaseAgent',
    'ConfigManager',
    'ExtractionAgent',
    'KnowledgeArchitectAgent',
    'InferenceAgent',
    'CoordinatorAgent',
]

__version__ = '1.0.0'

