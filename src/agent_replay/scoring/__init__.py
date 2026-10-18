"""Layered scoring: deterministic checks, behavioural invariants and an optional LLM judge."""

from agent_replay.scoring.invariants import InvariantSyntaxError, Rule, evaluate, parse_rules
from agent_replay.scoring.judge import JudgeConfig, JudgeVerdict
from agent_replay.scoring.scorecard import (
    CheckConfig,
    Finding,
    Layer,
    LayerResult,
    Scorecard,
    ScoringSpec,
    Verdict,
    score,
)

__all__ = [
    "CheckConfig",
    "Finding",
    "InvariantSyntaxError",
    "JudgeConfig",
    "JudgeVerdict",
    "Layer",
    "LayerResult",
    "Rule",
    "Scorecard",
    "ScoringSpec",
    "Verdict",
    "evaluate",
    "parse_rules",
    "score",
]
