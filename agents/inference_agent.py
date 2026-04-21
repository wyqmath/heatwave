"""
inference_agent.py - Inference Agent
======================================

Responsible for the final two stages of the HeDA protocol:

  Stage 6 – Vector Embedding
      Script: 9_multihop_generate.py
      Generate the Multi-Hop LBD Benchmark (N=4,000 questions) from
      the hold-out literature set (>=2025), balanced across four answer
      choices and cross-layer / within-layer categories.

  Stage 7 – Reasoning & Verification
      Scripts: 10_multihop_evaluate.py, 11_traceability_showcase.py,
               12_data_availability.py, 13_figure_generation.py
      Evaluate four LLMs under {No-KG, +KG} conditions (Table 1),
      generate provenance visualisation (Figure 6 / Table S1),
      produce Dataset S1, and render all data-driven figures.
"""

import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent, Stage, AgentState


class InferenceAgent(BaseAgent):
    """Runs Stage 6 (benchmark generation) and Stage 7 (evaluation & figures)."""

    STAGE_6_SCRIPTS = [
        "9_multihop_generate.py",
    ]
    STAGE_7_SCRIPTS = [
        "10_multihop_evaluate.py",
        "11_traceability_showcase.py",
        "12_data_availability.py",
        "13_figure_generation.py",
    ]

    def __init__(self):
        super().__init__(
            name="InferenceAgent",
            stages=[Stage.VECTOR_EMBEDDING, Stage.REASONING_VERIFICATION],
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
        self.logger.info("InferenceAgent started  (Stage 6-7)")

        results: Dict[str, Any] = {}

        r6 = self._run_stage(Stage.VECTOR_EMBEDDING, self.STAGE_6_SCRIPTS)
        results["stage_6"] = r6
        if not r6["success"]:
            self.state = AgentState.FAILED
            return {"success": False, "agent": self.name, "results": results}

        r7 = self._run_stage(Stage.REASONING_VERIFICATION, self.STAGE_7_SCRIPTS)
        results["stage_7"] = r7

        ok = r6["success"] and r7["success"]
        self.state = AgentState.SUCCESS if ok else AgentState.FAILED
        self.logger.info("InferenceAgent finished  [%s]", "OK" if ok else "FAIL")
        return {"success": ok, "agent": self.name, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = InferenceAgent()
    print(agent.run())
