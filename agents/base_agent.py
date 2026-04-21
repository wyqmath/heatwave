"""
base_agent.py - Base Agent Class & Configuration Manager
=========================================================

Provides shared infrastructure used by every HeDA agent:
  - ConfigManager : YAML configuration loader (multi-path search).
  - Stage         : Enum of the seven-stage protocol.
  - AgentState    : Runtime state machine.
  - BaseAgent     : Common logging, state tracking, and subprocess helpers.
"""

import os
import sys
import subprocess
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime


# ============================================================
#  Configuration Manager
# ============================================================

class ConfigManager:
    """Loads config/default.yaml and exposes it as a dict."""

    _config: Optional[Dict] = None
    _config_path: Optional[Path] = None

    SEARCH_PATHS = [
        Path("config/default.yaml"),
        Path("../config/default.yaml"),
        Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "config" / "default.yaml",
    ]

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict:
        if cls._config is not None and config_path is None:
            return cls._config

        if config_path:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            cls._config_path = path
            with open(path, "r", encoding="utf-8") as f:
                cls._config = yaml.safe_load(f) or {}
            return cls._config

        for search_path in cls.SEARCH_PATHS:
            resolved = search_path.resolve()
            if resolved.exists():
                cls._config_path = resolved
                with open(resolved, "r", encoding="utf-8") as f:
                    cls._config = yaml.safe_load(f) or {}
                return cls._config

        cls._config = {}
        return cls._config

    @classmethod
    def get(cls, key_path: str, default=None):
        """Dot-separated lookup, e.g. ``ConfigManager.get('llm.model')``."""
        config = cls.load_config()
        value = config
        for k in key_path.split("."):
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    @classmethod
    def reset(cls):
        cls._config = None
        cls._config_path = None


# ============================================================
#  Stage & State Enums
# ============================================================

class Stage(Enum):
    """HeDA Seven-Stage Protocol."""
    CORPUS_ACQUISITION      = ("Stage 1", "Corpus Acquisition & Filtering")
    ONTOLOGICAL_EXTRACTION  = ("Stage 2", "Ontological Extraction")
    SEMANTIC_DISAMBIGUATION = ("Stage 3", "Semantic Disambiguation")
    ATTRIBUTE_ENRICHMENT    = ("Stage 4", "Attribute Enrichment")
    TOPOLOGICAL_CONSTRUCTION = ("Stage 5", "Topological Construction")
    VECTOR_EMBEDDING        = ("Stage 6", "Vector Embedding")
    REASONING_VERIFICATION  = ("Stage 7", "Reasoning & Verification")

    def __init__(self, stage_id: str, description: str):
        self.stage_id = stage_id
        self.description = description


class AgentState(Enum):
    IDLE    = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED  = "failed"


# ============================================================
#  BaseAgent
# ============================================================

class BaseAgent:
    """Common base for all HeDA agents.

    Subclasses override :meth:`run` and call :meth:`_run_script` to
    delegate work to the numbered Python scripts in ``code/``.
    """

    CODE_DIR = Path(__file__).resolve().parent.parent / "code"

    def __init__(self, name: str, stages: List[Stage]):
        self.name = name
        self.stages = stages
        self.state = AgentState.IDLE
        self.config = ConfigManager.load_config()
        self.execution_log: List[Dict] = []

        self.logger = logging.getLogger(f"HeDA.{name}")
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                f"%(asctime)s [%(levelname)s] [{name}] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    # ---- stage bookkeeping ----

    def _log_stage_start(self, stage: Stage):
        self.execution_log.append({
            "stage": stage.stage_id,
            "description": stage.description,
            "status": "started",
            "start_time": datetime.now().isoformat(),
        })
        self.logger.info("=" * 60)
        self.logger.info(">>> %s: %s", stage.stage_id, stage.description)
        self.logger.info("=" * 60)

    def _log_stage_end(self, stage: Stage, success: bool, message: str = ""):
        status = "success" if success else "failed"
        for entry in reversed(self.execution_log):
            if entry["stage"] == stage.stage_id and entry["status"] == "started":
                entry["status"] = status
                entry["end_time"] = datetime.now().isoformat()
                entry["message"] = message
                break
        icon = "OK" if success else "FAIL"
        self.logger.info("[%s] %s: %s  %s", icon, stage.stage_id,
                         stage.description, message)

    # ---- script runner ----

    def _run_script(self, script_name: str) -> subprocess.CompletedProcess:
        """Run a script from ``code/`` as a subprocess.

        Returns the :class:`subprocess.CompletedProcess` so the caller
        can inspect *returncode*, *stdout*, and *stderr*.
        """
        script_path = self.CODE_DIR / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        self.logger.info("  Running %s ...", script_name)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(self.CODE_DIR),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self.logger.error("  %s failed (exit %d):\n%s",
                              script_name, result.returncode,
                              result.stderr[-500:] if result.stderr else "")
        else:
            self.logger.info("  %s completed successfully.", script_name)
        return result

    # ---- interface ----

    def run(self) -> Dict[str, Any]:
        """Execute all stages owned by this agent. Subclasses must override."""
        raise NotImplementedError

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "stages": [s.stage_id for s in self.stages],
            "execution_log": self.execution_log,
        }

    def __repr__(self):
        ids = ", ".join(s.stage_id for s in self.stages)
        return f"<{self.name} [{ids}] state={self.state.value}>"
