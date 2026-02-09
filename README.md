# AI-Driven Discovery of Bio-Ecological Mediation in Cascading Heatwave Risks

## Abstract

Compound heatwaves increasingly trigger complex cascading failures that propagate through interconnected physical and human systems, yet the fragmentation of disciplinary knowledge hinders the comprehensive mapping of these systemic risk topologies. This study introduces the **Heatwave Discovery Agent (HeDA)** as an autonomous scientific synthesis framework designed to bridge cognitive gaps by constructing a high-fidelity knowledge graph from **8,111 academic publications**. By structuring **70,297 evidence nodes**, the system exhibits enhanced inferential fidelity in capturing long-tail risk mechanisms and achieves a significant accuracy margin compared to standard foundation models including GPT-5.2 and Claude Sonnet 4.5 in complex reasoning tasks. The resulting topological analysis reveals a critical **bio-ecological mediation effect** where biological systems function as the primary non-linear amplifiers of thermal stress that transform physical meteorological hazards into systemic socioeconomic losses. We further identify latent functional couplings between theoretically distinct sectors such as the heat-induced synchronization of power grid failures and emergency medical capacity saturation. These findings elucidate the dynamics of compound climate risks and provide an empirical basis for shifting adaptation strategies from static sectoral defense to dynamic cross-system resilience.

## Framework

<p align="center">
  <img src="HeDA.png" alt="HeDA Framework" width="100%">
</p>

## Project Structure

```
├── agents/                          # Multi-Agent System
│   ├── __init__.py                  # Package initialization
│   ├── base_agent.py                # Base agent class & configuration
│   ├── coordinator_agent.py         # Central Coordination Unit (CCU)
│   ├── extraction_agent.py          # Extraction Agent (Stage 1-2)
│   ├── knowledge_architect_agent.py # Knowledge Architect Agent (Stage 3-5)
│   └── inference_agent.py           # Inference Agent (Stage 6-7)
├── code/                            # Core Processing Scripts
│   ├── 1_data_to_json.py            # Corpus acquisition & LLM triplet extraction
│   ├── 2_deal_json.py               # JSON normalization & validation
│   ├── 3_graph_enhancement.py       # Layer classification & attribute enrichment
│   ├── 4_com_nodes.py               # Unique node collection
│   ├── 5_nodes_group.py             # Semantic similarity grouping (FAISS)
│   ├── 6_clean_and_map_nodes.py     # LLM-based standardized naming
│   ├── 7_standardized_json.py       # Apply standardized node names
│   ├── 8_upload_neo4j.py            # Neo4j graph database upload
│   ├── 9_all_nodes.py               # Node list & adjacency generation
│   ├── 10_node_recommender.py       # Vector embedding & FAISS index
│   ├── 11_data_to_qa_by_hop.py      # Multi-hop QA dataset generation
│   ├── 12_balance_answer.py         # Answer distribution balancing
│   ├── 13_kgqa_evaluation.py        # KG-augmented QA evaluation
│   ├── 14_ablation_no_kg.py         # Ablation study (no-KG baseline)
│   ├── 15_multi_hop_reasoning.py    # Multi-hop reasoning engine
│   ├── 16_advanced_reasoning.py     # Cross-layer & novelty analysis
│   ├── 17_large_scale_mining.py     # Large-scale risk path mining
│   └── 18_anaysis.py               # Result analysis & visualization
├── config/
│   └── default.yaml                 # Default configuration
└── README.md
```

## Seven-Stage Protocol

HeDA implements a structured seven-stage protocol for knowledge graph construction and reasoning:

| Stage | Name | Description | Scripts |
|:-----:|------|-------------|---------|
| 1 | **Corpus Acquisition** | Parse 8,111 WOS publications, extract title/abstract, LLM-based triplet extraction | `1_data_to_json.py` |
| 2 | **Ontological Extraction** | JSON key normalization, structure validation, triplet standardization | `2_deal_json.py` |
| 3 | **Semantic Disambiguation** | Collect unique nodes, SentenceTransformer + FAISS semantic grouping, LLM standardized naming | `4_com_nodes.py` `5_nodes_group.py` `6_clean_and_map_nodes.py` |
| 4 | **Attribute Enrichment** | Layer classification (physical/social/economic), relation typing, confidence scoring | `3_graph_enhancement.py` `7_standardized_json.py` |
| 5 | **Topological Construction** | Batch MERGE upload to Neo4j graph database with indexing | `8_upload_neo4j.py` |
| 6 | **Vector Embedding** | Node list generation, DashScope embedding API, FAISS index construction | `9_all_nodes.py` `10_node_recommender.py` |
| 7 | **Reasoning & Verification** | Multi-hop QA, KGQA evaluation, cross-layer analysis, novelty scoring, large-scale mining | `11-18_*.py` |

## Multi-Agent Architecture

The system is orchestrated by four collaborative agents:

- **CoordinatorAgent** — Central Coordination Unit managing dynamic task scheduling, state monitoring, and error recovery across the full pipeline.
- **ExtractionAgent** — Handles Stage 1–2: corpus parsing, LLM-driven triplet extraction, and JSON normalization.
- **KnowledgeArchitectAgent** — Handles Stage 3–5: semantic disambiguation, attribute enrichment, and Neo4j topological construction.
- **InferenceAgent** — Handles Stage 6–7: vector embedding, multi-hop reasoning, cross-layer analysis, and large-scale risk path mining.

## Installation

```bash
pip install openai neo4j sentence-transformers faiss-cpu networkx numpy tqdm pyyaml
```

## Usage

**Run the full pipeline:**

```bash
python -m agents.coordinator_agent
```

**Run a specific agent only:**

```bash
python -m agents.coordinator_agent --agent extraction
python -m agents.coordinator_agent --agent knowledge_architect
python -m agents.coordinator_agent --agent inference
```

**Resume from a specific stage:**

```bash
python -m agents.coordinator_agent --start-from knowledge_architect
```

**Continue on failure:**

```bash
python -m agents.coordinator_agent --skip-on-fail
```

**Run individual scripts directly:**

```bash
cd code
python 1_data_to_json.py
python 2_deal_json.py
# ... and so on
```

## Configuration

Edit `config/default.yaml` to customize LLM, Neo4j, and embedding settings:

```yaml
llm:
  model: "qwen3-max"
  api_key: "your-api-key"

neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "your-password"

embedding:
  model: "text-embedding-v4"
  dimension: 1024
```

## Key Metrics

- **Novelty Score**: `NoveltyScore(P) = α·LF(P) + β·CLC(P) + γ·IP(P)` where α=0.5, β=0.3, γ=0.2
  - **LF** — Path Length Factor
  - **CLC** — Cross-Layer Complexity
  - **IP** — Information Payload
