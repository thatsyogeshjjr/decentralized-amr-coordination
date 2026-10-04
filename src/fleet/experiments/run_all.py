"""Run (resumably) and analyse the experiment campaigns in order, then regenerate the V&V report.

    python -m fleet.experiments.run_all [names...]
"""
from __future__ import annotations

import sys

from fleet.experiments.analysis import analyze
from fleet.experiments.runner import RESULTS, run_campaign

ORDER = [("topology", "stop_wait"), ("comms", "lat0_loss0"), ("blockage", "stop_wait"), ("density", "stop_wait"),
         ("acceptance", "stop_wait"), ("scale", "stop_wait"),
         # v3 campaigns (design doc v3); optb_data must precede training, advisor_ablation needs the trained model
         ("comms_v3", "nominal"), ("optb_data", None), ("advisor_ablation", "pibt")]
LABEL_ORDER = {"acceptance": ["stop_wait", "central", "pibt", "pibt_learned"]}


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or [n for n, _ in ORDER]
    for name, base in ORDER:
        if name not in names:
            continue
        print(f"=== campaign {name}", flush=True)
        run_campaign(f"configs/experiments/{name}.toml")
        analyze(RESULTS / name, base, LABEL_ORDER.get(name))
        print(f"=== analysed {name}", flush=True)
    from fleet.experiments import report
    report.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
