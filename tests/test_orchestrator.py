"""v3 FR-13 / acceptance gate 7: the experiment orchestrator configures and records runs but has no control path into
the fleet (preflight P5).  Layer 1: static dependency check.  Layer 2: the agents' runtime input boundary is enforced
and observed in a real visual run launched by the orchestrator, and injected decision fields have no effect."""
import ast
import json
from pathlib import Path

import pytest

from fleet.agent.simlink import SENSE_KEYS, SimLink

SRC = Path(__file__).resolve().parents[1] / "src" / "fleet"
ORCH = sorted((SRC / "experiments").glob("*.py")) + [SRC / "launch.py"]
FORBIDDEN = ("fleet.agent", "fleet.comms", "fleet.benchmark", "zmq")


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


@pytest.mark.parametrize("path", ORCH, ids=lambda p: p.name)
def test_orchestrator_modules_have_no_control_dependencies(path):
    bad = [m for m in imports(path) if any(m == f or m.startswith(f + ".") for f in FORBIDDEN)]
    assert not bad, f"{path.name} imports {bad}"


def test_simlink_rejects_foreign_types_and_strips_decision_fields():
    link = SimLink.__new__(SimLink)
    from collections import Counter
    link.accepted, link.rejected = Counter(), Counter()
    assert link.validate({"type": "set_route", "path": [1, 2]}) is None
    assert link.validate({"type": "assign", "task": "T0"}) is None
    m = link.validate({"type": "sense", "tick": 1, "t": 0.05, "pose": [0, 0, 0], "route": [3], "priority": 9e9})
    assert set(m) <= SENSE_KEYS and "route" not in m and "priority" not in m
    s = link.stats()
    assert s["rejected"] == {"type:set_route": 1, "type:assign": 1, "key:route": 1, "key:priority": 1}
    assert s["accepted"] == {"sense": 1}


def _inline(v):
    """TOML inline value for the injected test fields."""
    if isinstance(v, dict):
        return "{ " + ", ".join(f'"{k}" = {_inline(x)}' for k, x in v.items()) + " }"
    return json.dumps(v)


@pytest.mark.sim
def test_runtime_boundary_and_injection_has_no_effect(tmp_path):
    from fleet.experiments.runner import run_campaign
    from fleet.experiments.regression import fingerprint
    res = {}
    for name, inject in (("plain", None), ("inject", {"route": [1, 2, 3], "priority": 1e9, "winner": "amr2"})):
        p = tmp_path / f"{name}.toml"
        extra = f', "run.inject_sense" = {_inline(inject)}' if inject else ""
        p.write_text(f'''[campaign]
name = "{name}"
scenario = "configs/scenarios/m4_tasks.toml"
seeds = [2]
parallel = 1

[[condition]]
label = "base"
set = {{ "tasks.count" = 5, "run.duration" = 500.0{extra} }}
''')
        runs = run_campaign(p, root=tmp_path / "out")
        assert [r["status"] for r in runs] == ["ok"]
        d = tmp_path / "out" / name / "base" / "seed_2"
        res[name] = (json.loads((d / "result.json").read_text()), json.loads((d / "manifest.json").read_text()))
    r, man = res["plain"]
    # provenance record written by the orchestrator / launcher
    for k in ("config_hash", "code_version", "fault_schedule", "scenario", "seed", "start_wall", "end_wall", "attempts"):
        assert man.get(k) is not None, k
    assert man["status"] == "finished" and man["complete"] and man["attempts"][-1]["status"] == "ok"
    # every agent saw only whitelisted input from outside its peer network, and only peer-robot traffic
    for a in r["agents"].values():
        assert a["link"]["link_rejected"] == 0 and set(a["link"]["accepted"]) <= {"sense", "stop", "kill"}
        assert a["net"]["foreign_src"] == 0 and a["net"]["decode_errors"] == 0
    # injected route / priority / winner fields are stripped and do not change any decision
    ri, _ = res["inject"]
    for a in ri["agents"].values():
        assert a["link"]["rejected"] == {"key:route": a["link"]["accepted"]["sense"],
                                         "key:priority": a["link"]["accepted"]["sense"],
                                         "key:winner": a["link"]["accepted"]["sense"]}
    assert fingerprint(ri) == fingerprint(r)
    assert r["tasks"]["n_delivered"] == r["tasks"]["n_tasks"] == 5
