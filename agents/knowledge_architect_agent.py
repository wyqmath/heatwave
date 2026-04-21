"""
knowledge_architect_agent.py - Knowledge Architect Agent
=========================================================

Responsible for the middle three stages of the HeDA protocol:

  Stage 3 – Semantic Disambiguation
      Script: 6_entity_merge_eval.py
      Evaluate entity deduplication at multiple distance thresholds,
      select the optimal merge threshold, and measure precision via
      LLM-as-judge.

  Stage 4 – Attribute Enrichment
      Script: 7_topology_sensitivity.py
      Build the four-layer risk topology from merged KG JSONs, compute
      NoveltyScore sensitivity across 55 parameter combinations, and run
      a permutation test to validate the Bio-mediation Ratio (BMR).

  Stage 5 – Topological Construction
      Script: 8_layer_sensitivity.py
      Conduct three layer-merge sensitivity experiments (layer collapse/
      split, literature-volume normalisation, alternative layer rules)
      to verify BMR robustness.
"""

import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent, Stage, AgentState


class KnowledgeArchitectAgent(BaseAgent):
    """Runs Stage 3 (entity merge), Stage 4 (topology), and Stage 5 (layer sensitivity)."""

    STAGE_3_SCRIPTS = ["6_entity_merge_eval.py"]
    STAGE_4_SCRIPTS = ["7_topology_sensitivity.py"]
    STAGE_5_SCRIPTS = ["8_layer_sensitivity.py"]

    def __init__(self):
        super().__init__(
            name="KnowledgeArchitectAgent",
            stages=[
                Stage.SEMANTIC_DISAMBIGUATION,
                Stage.ATTRIBUTE_ENRICHMENT,
                Stage.TOPOLOGICAL_CONSTRUCTION,
            ],
        )

    def _run_stage(self, stage: Stage, scripts: List[str]) -> Dict[str, Any]:
        self._log_stage_start(stage)
        for script in scripts:
            result = self._run_script(script)
            if result.returncode != 0:
                msg = f"{script} exited with code {result.returncode}"
                self._log_stage_end(stage, False, msg)
                return {"success": False, "failed_script": script, "message": msg}
        self._log_stage_end(stage, True)
        return {"success": True}

    def run(self) -> Dict[str, Any]:
        self.state = AgentState.RUNNING
        self.logger.info("KnowledgeArchitectAgent started  (Stage 3-5)")

        results: Dict[str, Any] = {}

        for label, stage, scripts in [
            ("stage_3", Stage.SEMANTIC_DISAMBIGUATION, self.STAGE_3_SCRIPTS),
            ("stage_4", Stage.ATTRIBUTE_ENRICHMENT,    self.STAGE_4_SCRIPTS),
            ("stage_5", Stage.TOPOLOGICAL_CONSTRUCTION, self.STAGE_5_SCRIPTS),
        ]:
            r = self._run_stage(stage, scripts)
            results[label] = r
            if not r["success"]:
                self.state = AgentState.FAILED
                return {"success": False, "agent": self.name, "results": results}

        self.state = AgentState.SUCCESS
        self.logger.info("KnowledgeArchitectAgent finished  [OK]")
        return {"success": True, "agent": self.name, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = KnowledgeArchitectAgent()
    print(agent.run())
