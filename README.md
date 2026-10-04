# Decentralized AMR Fleet — MuJoCo + ZeroMQ

This is a simulation-only implementation of *Edge-AI Based Distributed Fleet Coordination for Autonomous Mobile Robots in Smart Warehouses*. The current design baseline is `docs/warehouse_system_design_doc_v3.pdf`; the original was `docs/warehouse_system_design_doc_v2.pdf`.

What it includes:
- MiR100 robots in a visual MuJoCo warehouse, one process per robot.
- Brokerless ZeroMQ peer messaging with a simulation-side impairment layer.
- Peer-to-peer task auctions.
- A* plus asynchronous PIBT.
- A deterministic safety shield.
- Blockage, deadlock and communication-isolation recovery.
- Optional priority advisors:
  - the historical v2 MLP;
  - the v3 Option B congestion-aware advisor.
- An observation-only dashboard.
- A non-controlling experiment orchestrator.

There is no ROS 2, no central runtime controller and no central inference.

## Documents
- Architecture: `docs/architecture.md`.
- Requirements traceability: `docs/requirements.md`.
- V&V report, generated from real runs: `docs/vv_report.md`. Sections 1–7 are the v2 evidence; section 8 is the v3 evidence.
- v3 preflight record and Option B specification: `docs/internal_updates/`.

## Setup

```
pip install -r requirements.txt
pip install -e .
```

The MiR100 meshes and URDF parameters are in `assets/mir100` (DFKI `mir_robot`, BSD-3).

## Run

Every MuJoCo run opens the viewer. There are two modes:
- `accelerated` (default): steps as fast as the robots respond. Used for data.
- `demo`: runs at 1× for presentations.

```
# pitch demo: 1x, dashboard at http://127.0.0.1:8050, pallet blockage at 60 s, robot crash at 150 s
python -m fleet.launch --scenario configs/scenarios/demo_warehouse.toml

# any scenario, accelerated; overrides with --set key=value
python -m fleet.launch --scenario configs/scenarios/exp_overlap.toml --set fleet.planner=stop_wait --seed 3

# v3 communication faults: temporary disconnection, carrier reconciliation
python -m fleet.launch --scenario configs/scenarios/v3_disconnect.toml
python -m fleet.launch --scenario configs/scenarios/v3_carrier_disconnect.toml
python -m fleet.launch --scenario configs/scenarios/exp_overlap.toml --set 'comms.burst={"p_enter":0.0222,"p_exit":0.2}' --set comms.jitter=0.1
```

### Planners
- `pibt`: the proposed decentralized planner.
- `stop_wait`: the baseline.
- `central`: a benchmark-only reference.

### Priority advisors (PIBT key only; they never command motion)
| Setting | Configuration |
|---|---|
| `fleet.priority_advisor=none` | A, deterministic |
| `fleet.priority_advisor=heuristic` | B, handcrafted congestion heuristic |
| `fleet.priority_advisor=learned` | C, Option B model; needs `models/optb_advisor.pt` |
| `fleet.learned_priority=true` | historical v2 MLP |

### Communication impairment (`comms.*`, simulation time)
- `latency`, `jitter`, `loss`.
- `burst`: Gilbert–Elliott `{p_enter, p_exit, loss_bad}`.
- `disconnect`: `[{robot, t, duration}]`.
- `reorder_pub`.
- `trace`: per-message CSV.

Delivery happens on 50 ms ticks.

## Tests

```
python -m pytest -m "not sim"     # unit + multi-process ZeroMQ tests (no simulator)
python -m pytest -m sim           # visual MuJoCo integration gates (accelerated)
python -m pytest --junitxml=results/pytest_v3.xml   # everything, report input (results/pytest.xml = v2 evidence)
python -m fleet.experiments.regression <tag>        # v2 behavioural regression (results/regression/<tag>)
```

## Learning

### Historical v2 advisor
```
python -m fleet.learning.train --out models/priority_mlp.pt --profile congested
```
CEM policy search over the eq. 7 MLP in the graph-level simulator.

### v3 Option B advisor
The specification is in `docs/internal_updates/option_b_spec.md`. The steps run in this order:
```
python -m fleet.learning.dataset splits                         # freeze seeds / layouts (before any data)
python -m fleet.experiments.runner configs/experiments/optb_data.toml   # 210 visual MuJoCo runs, policy A + feature logs
python -m fleet.learning.dataset build                          # blocking-impact labels, train/val vs test files
python -m fleet.learning.train_optb train                       # model selection on validation only
python -m fleet.learning.train_optb evaluate                    # one-time held-out test evaluation (freezes the model)
```

## Experiments

```
python -m fleet.experiments.run_all [campaign ...]   # run (resumable) + analyse + regenerate docs/vv_report.md
python -m fleet.experiments.runner configs/experiments/acceptance.toml
python -m fleet.experiments.analysis results/campaigns/acceptance --baseline stop_wait
python -m fleet.experiments.report       # -> docs/vv_report.md
```

### Campaigns
| Version | Campaigns |
|---|---|
| v2 (historical) | `acceptance` (App. D configurations), `topology`, `density`, `blockage`, `comms` (latency × loss), `scale` (N = 5, 8) |
| v3 | `comms_v3` (jitter, burst loss, stale state, disconnects), `optb_data` (Option B dataset), `advisor_ablation` (A / B / C with stop-and-wait and central references, held-out seeds and layout) |

Runs are resumable. Each run stores a `manifest.json` with:
- scenario, seed and config hash;
- code and runtime-code version;
- model version;
- the fault schedule;
- status and every attempt.

It also keeps the resolved config and raw logs.
