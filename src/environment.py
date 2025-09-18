# environment.py
from __future__ import annotations
from typing import Protocol, TypedDict, Literal, Optional, Mapping, Any
from dataclasses import dataclass
import asyncio
import time
import random

# ---------- Agent-facing types (stable contract) ----------

class Percept(TypedDict, total=False):
    obs: dict              # observations sufficient for policy decisions
    metrics: dict          # latency_ms, spend_usd, quality, errors, etc.
    events: list[dict]     # env decisions, governance overrides, logs

class Action(TypedDict):
    verb: Literal["PLAN","FETCH","OCR","EXTRACT","VALIDATE","RETRY","EXPLORE","NOOP"]
    params: dict

class EnvConfig(TypedDict, total=False):
    tool_allowlist: list[str]
    max_parallel: int
    spend_cap_usd: float
    seed: int
    policy_hint: dict

class Environment(Protocol):
    async def open(self, agent_id: str, cfg: EnvConfig) -> None: ...
    async def percept(self, agent_id: str) -> Percept: ...
    async def enact(self, agent_id: str, action: Action) -> Percept: ...
    async def close(self, agent_id: str) -> None: ...

# ---------- Tool adapters (actuators behind the action algebra) ----------

class ToolAdapter(Protocol):
    async def call(self, **kwargs) -> dict: ...

@dataclass
class FakeFetcher:
    """Toy fetcher; imagine this hits a queue or inbox API."""
    inbox: int = 12
    async def call(self, batch_size: int = 1, **_) -> dict:
        taken = min(self.inbox, batch_size)
        self.inbox -= taken
        await asyncio.sleep(0)
        return {"taken": taken, "remaining": self.inbox, "spend_usd": 0.01*taken, "quality": 0.0}

@dataclass
class FakeOCR:
    base_quality: float = 0.86
    async def call(self, **_) -> dict:
        await asyncio.sleep(0)
        jitter = random.uniform(-0.01, 0.03)
        q = max(0.0, min(1.0, self.base_quality + jitter))
        return {"quality": q, "spend_usd": 0.02}

@dataclass
class FakeExtract:
    async def call(self, mode: str = "normal", **_) -> dict:
        await asyncio.sleep(0)
        q = 0.88 if mode == "normal" else 0.90
        return {"quality": q, "spend_usd": 0.01}

# ---------- Runner/EnvSession (physics + orchestration) ----------

class RunnerEnv(Environment):
    """
    Orchestrates perception and actuation.
    Owns time/cost accounting, parallelism, and event logging.
    """
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    async def open(self, agent_id: str, cfg: EnvConfig) -> None:
        seed = cfg.get("seed", 0)
        random.seed(seed)
        self._sessions[agent_id] = {
            "cfg": cfg,
            "fetcher": FakeFetcher(),
            "ocr": FakeOCR(),
            "extract": FakeExtract(),
            "budget": cfg.get("spend_cap_usd", 10.0),
            "spent": 0.0,
            "events": [],
        }

    async def percept(self, agent_id: str) -> Percept:
        s = self._sessions[agent_id]
        return {
            "obs": {"inbox_count": s["fetcher"].inbox},
            "metrics": {"quality_p50": 0.85, "spend_usd": s["spent"], "errors": 0},
            "events": s["events"][-5:],  # last few events
        }

    async def enact(self, agent_id: str, action: Action) -> Percept:
        s = self._sessions[agent_id]
        t0 = time.time()
        verb = action["verb"]
        params = action.get("params", {})

        # Governance/allow-list check (example)
        allow = s["cfg"].get("tool_allowlist")
        if allow is not None and verb not in {"NOOP","PLAN","RETRY","EXPLORE"} | set(allow):
            ev = {"type": "governance_override", "verb": verb}
            s["events"].append(ev)
            return {"events": [ev], "metrics": {"errors": 1, "latency_ms": 0}}

        outcome: dict = {}
        if verb == "FETCH":
            outcome = await s["fetcher"].call(**params)
        elif verb == "OCR":
            outcome = await s["ocr"].call(**params)
        elif verb == "EXTRACT":
            outcome = await s["extract"].call(**params)
        elif verb in {"VALIDATE","RETRY","EXPLORE","PLAN","NOOP"}:
            await asyncio.sleep(0)
            outcome = {"quality": 0.0, "spend_usd": 0.0}
        else:
            outcome = {"errors": 1}

        spend = float(outcome.get("spend_usd", 0.0))
        s["spent"] += spend

        # Budget enforcement
        if s["spent"] > s["budget"]:
            ev = {"type": "budget_cap_exceeded", "spent": s["spent"], "cap": s["budget"]}
            s["events"].append(ev)

        latency_ms = int((time.time() - t0) * 1000)
        metrics = {
            "latency_ms": latency_ms,
            "spend_usd": spend,
            "quality": outcome.get("quality", 0.0),
            "errors": int(outcome.get("errors", 0)),
        }
        obs = {"inbox_count": s["fetcher"].inbox}
        return {"obs": obs, "metrics": metrics, "events": s["events"][-5:]}

    async def close(self, agent_id: str) -> None:
        self._sessions.pop(agent_id, None)
