"""
extraction_agent.py - Extraction Agent
=======================================

Responsible for the first two stages of the HeDA protocol:

  Stage 1 – Corpus Acquisition & Filtering
      Scripts: 1_data_to_json.py
      Extract causal triples from WoS abstracts using an LLM with a
      four-layer ontology (Physical / Biological / Social / Economic).

  Stage 2 – Ontological Extraction
      Scripts: 2_sample_fulltext.py, 3_pdf_to_markdown.py,
               4_fulltext_extract.py, 5_fulltext_compare.py
      Validate extraction quality via a full-text comparison experiment:
      stratified-sample 80 OA papers, convert PDFs to Markdown, extract
      triples from full text, and compare against abstract-only results.
"""

import logging
from typing import Dict, Any, List

from agents.base_agent import BaseAgent, Stage, AgentState


class ExtractionAgent(BaseAgent):
    """Runs Stage 1 (abstract extraction) and Stage 2 (full-text validation)."""

    STAGE_1_SCRIPTS = [
        "1_data_to_json.py",
    ]
    STAGE_2_SCRIPTS = [
        "2_sample_fulltext.py",
        "3_pdf_to_markdown.py",
        "4_fulltext_extract.py",
        "5_fulltext_compare.py",
    ]

    def __init__(self):
        super().__init__(
            name="ExtractionAgent",
            stages=[Stage.CORPUS_ACQUISITION, Stage.ONTOLOGICAL_EXTRACTION],
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
        self.logger.info("ExtractionAgent started  (Stage 1-2)")

        results: Dict[str, Any] = {}

        r1 = self._run_stage(Stage.CORPUS_ACQUISITION, self.STAGE_1_SCRIPTS)
        results["stage_1"] = r1
        if not r1["success"]:
            self.state = AgentState.FAILED
            return {"success": False, "agent": self.name, "results": results}

        r2 = self._run_stage(Stage.ONTOLOGICAL_EXTRACTION, self.STAGE_2_SCRIPTS)
        results["stage_2"] = r2

        ok = r1["success"] and r2["success"]
        self.state = AgentState.SUCCESS if ok else AgentState.FAILED
        self.logger.info("ExtractionAgent finished  [%s]", "OK" if ok else "FAIL")
        return {"success": ok, "agent": self.name, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = ExtractionAgent()
    print(agent.run())
