"""Baseline suites: a named, versioned set of recorded runs that a candidate is gated against.

On disk a suite is a directory with a ``suite.json`` manifest and one trajectory file per case::

    suites/support-desk-v1/
      suite.json
      cases/refund-damaged.json
      ...

The manifest pins each case's behavioural fingerprint. ``load_suite`` refuses a suite whose
files no longer match it, so an edited or truncated baseline cannot silently change what the
gate compares against. Suites can also be pushed to the store and read back over the API.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from agent_replay.agent import Agent
from agent_replay.clock import TickingClock
from agent_replay.errors import AgentReplayError
from agent_replay.model import Trajectory
from agent_replay.recorder import record_run

MANIFEST = "suite.json"
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class SuiteError(AgentReplayError):
    """A suite is missing, malformed or does not match its manifest."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SuiteCase(_Frozen):
    id: str
    task: str
    trajectory_id: str
    fingerprint: str
    file: str | None = None  # relative to the suite directory; None when stored in a database

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError(f"case id {value!r} must match {_ID.pattern}")
        return value


class SuiteManifest(_Frozen):
    name: str
    description: str = ""
    agent_name: str
    created_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    cases: tuple[SuiteCase, ...]

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError(f"suite name {value!r} must match {_ID.pattern}")
        return value

    @field_validator("cases")
    @classmethod
    def _unique(cls, cases: tuple[SuiteCase, ...]) -> tuple[SuiteCase, ...]:
        if not cases:
            raise ValueError("a suite needs at least one case")
        ids = [c.id for c in cases]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate case ids")
        return cases


class Suite(_Frozen):
    manifest: SuiteManifest
    baselines: dict[str, Trajectory]  # case id -> recorded baseline

    @property
    def name(self) -> str:
        return self.manifest.name

    def items(self) -> list[tuple[SuiteCase, Trajectory]]:
        return [(case, self.baselines[case.id]) for case in self.manifest.cases]


def verify(manifest: SuiteManifest, baselines: dict[str, Trajectory]) -> Suite:
    """Check every case has a baseline whose ID and fingerprint match the manifest."""
    for case in manifest.cases:
        trajectory = baselines.get(case.id)
        if trajectory is None:
            raise SuiteError(f"suite {manifest.name}: case {case.id} has no baseline")
        if trajectory.trajectory_id != case.trajectory_id:
            raise SuiteError(
                f"suite {manifest.name}: case {case.id} expected trajectory "
                f"{case.trajectory_id}, found {trajectory.trajectory_id}"
            )
        if trajectory.fingerprint() != case.fingerprint:
            raise SuiteError(
                f"suite {manifest.name}: case {case.id} baseline does not match the "
                "fingerprint in the manifest (edited or corrupted)"
            )
    extra = set(baselines) - {c.id for c in manifest.cases}
    if extra:
        raise SuiteError(f"suite {manifest.name}: baselines for unknown cases {sorted(extra)}")
    return Suite(manifest=manifest, baselines=baselines)


def check_stored(manifest: SuiteManifest, fingerprints: dict[str, str]) -> SuiteManifest:
    """Validate a manifest against the fingerprints of stored trajectories (by trajectory ID)
    and return it as stored: without file paths, which mean nothing inside a database."""
    for case in manifest.cases:
        stored = fingerprints.get(case.trajectory_id)
        if stored is None:
            raise SuiteError(
                f"suite {manifest.name}: trajectory {case.trajectory_id} is not stored"
            )
        if stored != case.fingerprint:
            raise SuiteError(
                f"suite {manifest.name}: stored trajectory {case.trajectory_id} does not match "
                "the fingerprint in the manifest"
            )
    return manifest.model_copy(
        update={"cases": tuple(c.model_copy(update={"file": None}) for c in manifest.cases)}
    )


class SuiteSource(Protocol):
    """The part of a trajectory store a stored suite is loaded from."""

    async def get_suite(self, name: str) -> SuiteManifest | None: ...

    async def get(self, trajectory_id: str) -> Trajectory | None: ...


async def load_stored_suite(source: SuiteSource, name: str) -> Suite:
    manifest = await source.get_suite(name)
    if manifest is None:
        raise SuiteError(f"no stored suite named {name!r}")
    baselines: dict[str, Trajectory] = {}
    for case in manifest.cases:
        trajectory = await source.get(case.trajectory_id)
        if trajectory is None:
            raise SuiteError(f"suite {name}: trajectory {case.trajectory_id} is not stored")
        baselines[case.id] = trajectory
    return verify(manifest, baselines)


def load_suite(directory: str | Path) -> Suite:
    root = Path(directory)
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise SuiteError(f"{root} is not a suite (no {MANIFEST})")
    manifest = SuiteManifest.model_validate_json(manifest_path.read_bytes())
    baselines: dict[str, Trajectory] = {}
    resolved_root = root.resolve()
    for case in manifest.cases:
        if case.file is None:
            raise SuiteError(f"case {case.id} has no file")
        path = (root / case.file).resolve()
        if not path.is_relative_to(resolved_root):
            raise SuiteError(f"case {case.id} points outside the suite directory")
        baselines[case.id] = Trajectory.from_json(path.read_bytes())
    return verify(manifest, baselines)


def write_suite(suite: Suite, directory: str | Path) -> Path:
    root = Path(directory)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    cases = []
    for case, trajectory in suite.items():
        relative = f"cases/{case.id}.json"
        (root / relative).write_text(trajectory.to_json(), encoding="utf-8")
        cases.append(case.model_copy(update={"file": relative}))
    manifest = suite.manifest.model_copy(update={"cases": tuple(cases)})
    (root / MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return root


def slug(text: str, index: int) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())[:5]
    return f"{index:02d}-" + "-".join(words) if words else f"{index:02d}"


async def record_suite(
    name: str,
    tasks: Sequence[str],
    agent_factory: Callable[[str], Agent],
    *,
    description: str = "",
    metadata: dict[str, JsonValue] | None = None,
    now: datetime | None = None,
) -> Suite:
    """Record one baseline per task. ``agent_factory`` receives the case ID."""
    cases: list[SuiteCase] = []
    baselines: dict[str, Trajectory] = {}
    agent_name = ""
    for index, task in enumerate(tasks):
        case_id = slug(task, index)
        agent = agent_factory(case_id)
        trajectory = await record_run(
            agent, task, clock=TickingClock(), trajectory_id=f"{name}/{case_id}"
        )
        agent_name = trajectory.agent_name
        baselines[case_id] = trajectory
        cases.append(
            SuiteCase(
                id=case_id,
                task=task,
                trajectory_id=trajectory.trajectory_id,
                fingerprint=trajectory.fingerprint(),
            )
        )
    manifest = SuiteManifest(
        name=name,
        description=description,
        agent_name=agent_name,
        created_at=now or datetime.now(UTC),
        metadata=metadata or {},
        cases=tuple(cases),
    )
    return verify(manifest, baselines)
