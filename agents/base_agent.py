#!/usr/bin/env python3
"""
base_agent.py - HeDA基础Agent类与配置管理器
=============================================

提供所有Agent的公共基类和统一配置管理：
  - ConfigManager: YAML配置加载器，支持多路径搜索
  - BaseAgent: Agent基类，提供日志、状态管理、LLM调用等公共能力
"""

import os
import sys
import json
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime


# ============================================================
#  配置管理器
# ============================================================

class ConfigManager:
    """统一配置管理器 - 加载YAML配置文件并提供全局访问"""

    _config: Optional[Dict] = None
    _config_path: Optional[Path] = None

    # 配置文件搜索路径（优先级从高到低）
    SEARCH_PATHS = [
        Path("config/default.yaml"),
        Path("../config/default.yaml"),
        Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "config" / "default.yaml",
    ]

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict:
        """加载配置文件
        
        Args:
            config_path: 指定配置文件路径，为None时自动搜索
            
        Returns:
            配置字典
        """
        if cls._config is not None and config_path is None:
            return cls._config

        if config_path:
            path = Path(config_path)
            if path.exists():
                cls._config_path = path
                with open(path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f) or {}
                return cls._config
            else:
                raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 自动搜索配置文件
        for search_path in cls.SEARCH_PATHS:
            resolved = search_path.resolve()
            if resolved.exists():
                cls._config_path = resolved
                with open(resolved, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f) or {}
                return cls._config

        # 未找到配置文件，返回默认配置
        cls._config = cls._get_default_config()
        return cls._config

    @classmethod
    def _get_default_config(cls) -> Dict:
        """返回默认配置"""
        return {
            'llm': {
                'api_key': os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE'),
                'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'model': 'qwen3-max',
                'max_tokens': 8000,
                'temperature': 0.7,
            },
            'embedding': {
                'model': 'text-embedding-v4',
                'dimension': 1024,
            },
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'user': 'neo4j',
                'password': os.environ.get('NEO4J_PASSWORD', ''),
            },
            'data_processing': {
                'input_file': 'paper.txt',
                'output_dir': 'json/',
                'enhanced_dir': 'enhanced_json/',
                'max_workers': 15,
            },
            'paths': {
                'json_dir': 'json/',
                'enhanced_json_dir': 'enhanced_json/',
                'reports_dir': 'reports/',
                'nodes_file': 'all_nodes.csv',
                'adjacency_file': 'adjacency.json',
                'mapping_file': 'mapping.csv',
            },
        }

    @classmethod
    def get(cls, key_path: str, default=None):
        """通过点分路径获取配置值，如 'llm.api_key'"""
        config = cls.load_config()
        keys = key_path.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @classmethod
    def reset(cls):
        """重置配置缓存"""
        cls._config = None
        cls._config_path = None


# ============================================================
#  Agent状态枚举
# ============================================================

class AgentState(Enum):
    """Agent运行状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    WAITING = "waiting"


# ============================================================
#  Stage定义
# ============================================================

class Stage(Enum):
    """HeDA七阶段协议"""
    CORPUS_ACQUISITION = ("Stage1", "Corpus Acquisition & Filtering", "语料获取与过滤")
    ONTOLOGICAL_EXTRACTION = ("Stage2", "Ontological Extraction", "本体抽取")
    SEMANTIC_DISAMBIGUATION = ("Stage3", "Semantic Disambiguation", "语义消歧")
    ATTRIBUTE_ENRICHMENT = ("Stage4", "Attribute Enrichment", "属性增强")
    TOPOLOGICAL_CONSTRUCTION = ("Stage5", "Topological Construction", "拓扑构建")
    VECTOR_EMBEDDING = ("Stage6", "Vector Embedding", "向量嵌入")
    REASONING_VERIFICATION = ("Stage7", "Reasoning & Verification", "推理与验证")

    def __init__(self, stage_id, name_en, name_cn):
        self.stage_id = stage_id
        self.name_en = name_en
        self.name_cn = name_cn


# ============================================================
#  BaseAgent 基类
# ============================================================

class BaseAgent:
    """所有HeDA Agent的基类

    提供公共能力：
      - 统一日志系统
      - 状态管理与追踪
      - LLM调用封装
      - 阶段执行记录
    """

    def __init__(self, name: str, stages: List[Stage]):
        """
        Args:
            name: Agent名称
            stages: 该Agent负责的阶段列表
        """
        self.name = name
        self.stages = stages
        self.state = AgentState.IDLE
        self.config = ConfigManager.load_config()
        self.execution_log: List[Dict] = []

        # 设置日志
        self.logger = logging.getLogger(f"HeDA.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                f'%(asctime)s [%(levelname)s] [{name}] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # 确保工作目录正确
        self._setup_directories()

    def _setup_directories(self):
        """确保必要的输出目录存在"""
        dirs = [
            self.config.get('paths', {}).get('json_dir', 'json/'),
            self.config.get('paths', {}).get('enhanced_json_dir', 'enhanced_json/'),
            self.config.get('paths', {}).get('reports_dir', 'reports/'),
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _log_stage_start(self, stage: Stage):
        """记录阶段开始"""
        entry = {
            'stage': stage.stage_id,
            'name': stage.name_en,
            'status': 'started',
            'start_time': datetime.now().isoformat(),
        }
        self.execution_log.append(entry)
        self.logger.info(f"{'='*60}")
        self.logger.info(f"▶ 开始执行 {stage.stage_id}: {stage.name_cn}({stage.name_en})")
        self.logger.info(f"{'='*60}")

    def _log_stage_end(self, stage: Stage, success: bool, message: str = ""):
        """记录阶段结束"""
        status = 'success' if success else 'failed'
        # 更新最后一条日志
        for entry in reversed(self.execution_log):
            if entry['stage'] == stage.stage_id and entry['status'] == 'started':
                entry['status'] = status
                entry['end_time'] = datetime.now().isoformat()
                entry['message'] = message
                break

        icon = "✔" if success else "✘"
        self.logger.info(f"{icon}{stage.stage_id}: {stage.name_cn} - {'成功' if success else '失败'}")
        if message:
            self.logger.info(f"  详情: {message}")

    def get_llm_config(self) -> Dict:
        """获取LLM配置"""
        return {
            'api_key': self.config.get('llm', {}).get('api_key', os.environ.get('HEDA_API_KEY', 'YOUR_API_KEY_HERE')),
            'base_url': self.config.get('llm', {}).get('base_url', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
            'model': self.config.get('llm', {}).get('model', 'qwen3-max'),
            'max_tokens': self.config.get('llm', {}).get('max_tokens', 8000),
            'temperature': self.config.get('llm', {}).get('temperature', 0.7),
        }

    def get_neo4j_config(self) -> Dict:
        """获取Neo4j配置"""
        return {
            'uri': self.config.get('neo4j', {}).get('uri', 'bolt://localhost:7687'),
            'user': self.config.get('neo4j', {}).get('user', 'neo4j'),
            'password': self.config.get('neo4j', {}).get('password', os.environ.get('NEO4J_PASSWORD', '')),
        }

    def get_embedding_config(self) -> Dict:
        """获取Embedding配置"""
        return {
            'model': self.config.get('embedding', {}).get('model', 'text-embedding-v4'),
            'dimension': self.config.get('embedding', {}).get('dimension', 1024),
        }

    def run(self) -> Dict[str, Any]:
        """执行Agent的所有阶段（子类必须实现）

        Returns:
            执行结果字典，包含 success, stages_completed, message 等
        """
        raise NotImplementedError("子类必须实现 run() 方法")

    def get_status(self) -> Dict:
        """获取Agent当前状态"""
        return {
            'name': self.name,
            'state': self.state.value,
            'stages': [s.stage_id for s in self.stages],
            'execution_log': self.execution_log,
        }

    def __repr__(self):
        stages_str = ", ".join(s.stage_id for s in self.stages)
        return f"<{self.name} [{stages_str}] state={self.state.value}>"

