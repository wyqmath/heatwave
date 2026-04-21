"""
HeDA (Heatwave Discovery Agent) - Multi-Agent Coordination System
=================================================================

A four-agent hierarchical system implementing a seven-stage neuro-symbolic
protocol for constructing and analysing heatwave cascading-risk knowledge
graphs from scientific literature.

Agents:
  - CoordinatorAgent:          Central Coordination Unit; orchestrates the
                               three worker agents in dependency order.
  - ExtractionAgent:           Corpus Acquisition & Filtering (Stage 1) +
                               Ontological Extraction (Stage 2).
  - KnowledgeArchitectAgent:   Semantic Disambiguation (Stage 3) +
                               Attribute Enrichment (Stage 4) +
                               Topological Construction (Stage 5).
  - InferenceAgent:            Vector Embedding (Stage 6) +
                               Reasoning & Verification (Stage 7).
"""

from agents.base_agent import BaseAgent, ConfigManager
from agents.extraction_agent import ExtractionAgent
from agents.knowledge_architect_agent import KnowledgeArchitectAgent
from agents.inference_agent import InferenceAgent
from agents.coordinator_agent import CoordinatorAgent

__all__ = [
    "BaseAgent",
    "ConfigManager",
    "ExtractionAgent",
    "KnowledgeArchitectAgent",
    "InferenceAgent",
    "CoordinatorAgent",
]

__version__ = "1.0.0"
