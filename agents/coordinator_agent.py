"""
coordinator_agent.py - Coordinator Agent (Central Coordination Unit)
=====================================================================

Orchestrates the three worker agents in strict dependency order:

  ExtractionAgent          (Stage 1-2)
    -> KnowledgeArchitectAgent  (Stage 3-5)
      -> InferenceAgent         (Stage 6-7)

Provides CLI entry-point for running the full pipeline or individual
agents via command-line flags.
"""

import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent, Stage, AgentState
from agents.extraction_agent import ExtractionAgent
from agents.knowledge_architect_agent import KnowledgeArchitectAgent
from agents.inference_agent import InferenceAgent


class CoordinatorAgent(BaseAgent):
    """Central Coordination Unit - manages the three worker agents."""

    AGENT_ORDER = ["extraction", "knowledge_architect", "inference"]

    def __init__(self):
        super().__init__(
            name="CoordinatorAgent",
            stages=list(Stage),
        )
        self.sub_agents: Dict[str, BaseAgent] = {}

    def _init_sub_agents(self):
        self.sub_agents = {
            "extraction": ExtractionAgent(),
            "knowledge_architect": KnowledgeArchitectAgent(),
            "inference": InferenceAgent(),
        }
        for agent in self.sub_agents.values():
            stages = ", ".join(s.stage_id for s in agent.stages)
            self.logger.info("  Registered %s  (%s)", agent.name, stages)

    def _execute_agent(self, key: str) -> Dict[str, Any]:
        agent = self.sub_agents[key]
        self.logger.info("-" * 60)
        self.logger.info(">>> Launching %s", agent.name)
        self.logger.info("-" * 60)

        t0 = time.time()
        try:
            result = agent.run()
        except Exception as e:
            result = {"success": False, "message": str(e)}
        result["elapsed_seconds"] = round(time.time() - t0, 2)

        status = "OK" if result.get("success") else "FAIL"
        self.logger.info("[%s] %s  (%.1fs)", status, agent.name,
                         result["elapsed_seconds"])
        return result

    # ---- public interface ----

    def run(self, *, skip_on_fail: bool = False,
            start_from: Optional[str] = None) -> Dict[str, Any]:
        """Run the full HeDA pipeline.

        Args:
            skip_on_fail: If True, continue to the next agent even when
                          the current one fails.
            start_from:   Start execution from a specific agent key
                          (``extraction`` / ``knowledge_architect`` /
                          ``inference``).
        """
        self.state = AgentState.RUNNING
        t0 = time.time()

        self.logger.info("=" * 60)
        self.logger.info("  HeDA Pipeline - Central Coordination Unit")
        self.logger.info("=" * 60)

        self._init_sub_agents()

        order = list(self.AGENT_ORDER)
        if start_from and start_from in order:
            order = order[order.index(start_from):]
            self.logger.info("Resuming from: %s", start_from)

        agent_results: Dict[str, Any] = {}
        overall = True

        for key in order:
            result = self._execute_agent(key)
            agent_results[key] = result
            if not result.get("success"):
                overall = False
                if not skip_on_fail:
                    self.logger.error("Pipeline halted at %s", key)
                    break

        elapsed = round(time.time() - t0, 2)
        self.state = AgentState.SUCCESS if overall else AgentState.FAILED

        self.logger.info("=" * 60)
        self.logger.info("  Pipeline %s  (%.1fs total)",
                         "SUCCEEDED" if overall else "FAILED", elapsed)
        self.logger.info("=" * 60)

        return {
            "success": overall,
            "agent_results": agent_results,
            "total_elapsed_seconds": elapsed,
        }

    def run_single(self, key: str) -> Dict[str, Any]:
        """Run only one sub-agent by key."""
        self._init_sub_agents()
        if key not in self.sub_agents:
            return {"success": False, "message": f"Unknown agent: {key}"}
        return self._execute_agent(key)


# ============================================================
#  CLI entry-point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeDA Pipeline Coordinator")
    parser.add_argument("--start-from", type=str, default=None,
                        choices=CoordinatorAgent.AGENT_ORDER,
                        help="Resume pipeline from a specific agent")
    parser.add_argument("--skip-on-fail", action="store_true",
                        help="Continue to next agent on failure")
    parser.add_argument("--agent", type=str, default=None,
                        choices=CoordinatorAgent.AGENT_ORDER,
                        help="Run a single agent only")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    coordinator = CoordinatorAgent()

    if args.agent:
        result = coordinator.run_single(args.agent)
    else:
        result = coordinator.run(
            skip_on_fail=args.skip_on_fail,
            start_from=args.start_from,
        )

    print(json.dumps(result, indent=2, default=str))
