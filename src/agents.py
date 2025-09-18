# agent.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional
from environment import Environment, Percept, Action, EnvConfig

# ---------- Internal state ----------

@dataclass
class Belief:
    backlog: int = 0
    ocr_quality_p50: float = 0.0
    spend_used: float = 0.0
    risk_level: str = "normal"
    last_error: Optional[str] = None

# ---------- Four AIMA-inspired components ----------

class PerformanceElement(Protocol):
    def decide(self, belief: Belief, percept: Percept) -> Action: ...

class Critic(Protocol):
    def evaluate(self, belief: Belief, p_before: Percept, a: Action, p_after: Percept) -> float: ...

class LearningElement(Protocol):
    def update(self, belief: Belief, reward: float, p_after: Percept) -> Belief: ...

class ProblemGenerator(Protocol):
    def propose(self, belief: Belief, reward: float) -> Optional[Action]: ...

# ---------- Simple defaults (generic) ----------

class SimplePolicy(PerformanceElement):
    def decide(self, belief: Belief, p: Percept) -> Action:
        obs = p.get("obs", {})
        metrics = p.get("metrics", {})
        belief.backlog = int(obs.get("inbox_count", 0))
        belief.ocr_quality_p50 = float(metrics.get("quality_p50", belief.ocr_quality_p50 or 0.85))

        if metrics.get("last_error"):
            return {"verb": "RETRY", "params": {"stage": "OCR"}}
        if belief.backlog > 0:
            return {"verb": "FETCH", "params": {"batch_size": min(5, belief.backlog)}}
        if belief.ocr_quality_p50 < 0.88:
            return {"verb": "EXTRACT", "params": {"mode": "strict"}}
        return {"verb": "NOOP", "params": {}}

class SimpleCritic(Critic):
    def evaluate(self, belief: Belief, p_before: Percept, a: Action, p_after: Percept) -> float:
        m = p_after.get("metrics", {})
        quality = float(m.get("quality", 0.0))
        latency = float(m.get("latency_ms", 0)) / 1000.0
        spend = float(m.get("spend_usd", 0.0))
        errors = int(m.get("errors", 0))
        return 1.5*quality - 0.5*latency - 0.2*spend - 1.0*errors

class SimpleLearner(LearningElement):
    def update(self, belief: Belief, reward: float, p_after: Percept) -> Belief:
        m = p_after.get("metrics", {})
        q = m.get("quality_p50")
        if isinstance(q, (int, float)):
            belief.ocr_quality_p50 = 0.9*belief.ocr_quality_p50 + 0.1*float(q)
        belief.spend_used += float(m.get("spend_usd", 0.0))
        belief.last_error = m.get("last_error")
        if reward < -0.5:
            belief.risk_level = "conservative"
        elif reward > 0.5:
            belief.risk_level = "normal"
        return belief

class SimpleExplorer(ProblemGenerator):
    def propose(self, belief: Belief, reward: float) -> Optional[Action]:
        if reward < 0.2 and belief.risk_level == "normal":
            return {"verb": "EXPLORE", "params": {"tool": "ocr_backup_provider"}}
        return None

# ---------- Agent process (pure brain; env stays outside) ----------

@dataclass
class Agent:
    agent_id: str
    policy: PerformanceElement
    critic: Critic
    learner: LearningElement
    explorer: ProblemGenerator
    belief: Belief = field(default_factory=Belief)

    async def episode(self, env: Environment, cfg: EnvConfig, steps: int = 5) -> None:
        await env.open(self.agent_id, cfg)
        try:
            for _ in range(steps):
                p_before = await env.percept(self.agent_id)
                act = self.policy.decide(self.belief, p_before)
                p_after = await env.enact(self.agent_id, act)
                r = self.critic.evaluate(self.belief, p_before, act, p_after)

                probe = self.explorer.propose(self.belief, r)
                if probe:
                    p_after = await env.enact(self.agent_id, probe)
                    r = self.critic.evaluate(self.belief, p_before, probe, p_after)

                self.belief = self.learner.update(self.belief, r, p_after)
        finally:
            await env.close(self.agent_id)
