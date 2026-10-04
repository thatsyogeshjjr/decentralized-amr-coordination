"""Seeded fault schedules used by the experiment matrix (blockage rate) are reproducible and nested across rates."""
from fleet.common.config import PROJECT_ROOT, load_config
from fleet.world.mapgrid import WarehouseMap
from fleet.world.sim_server import TaskBook, random_blockage_events

M = WarehouseMap.load(PROJECT_ROOT / "configs/maps/warehouse_small.txt")


def cfg(rate, seed=1):
    return load_config("configs/scenarios/exp_overlap.toml",
                       {"run": {"seed": seed}, "faults": {"random_blockages": {"rate": rate, "slots": 50}}})


def test_blockage_rate_and_reproducibility():
    assert random_blockage_events(cfg(0.0), M) == []
    a, b = random_blockage_events(cfg(0.2), M), random_blockage_events(cfg(0.2), M)
    assert a == b and 4 <= len(a) <= 18                      # ~20 % of 50 candidate events
    for e in a:
        n = e["node"][0] * M.W + e["node"][1]
        assert M.kind[n] == "aisle" and n not in M.starts.values()
    # common random numbers: every rate-10 % blockage also occurs at rate 20 %
    low = random_blockage_events(cfg(0.1), M)
    assert all(e in a for e in low)
    assert random_blockage_events(cfg(0.2, seed=2), M) != a


def test_order_stream_is_seeded():
    c = load_config("configs/scenarios/exp_overlap.toml", {"run": {"seed": 3}})
    o1 = [(o.task_id, o.release, o.pickup, o.drop) for o in TaskBook(c, M).orders]
    o2 = [(o.task_id, o.release, o.pickup, o.drop) for o in TaskBook(c, M).orders]
    assert o1 == o2 and len(o1) == 12
    assert all(M.kind[p] == "pickup" for _, _, p, _ in o1) and all(M.kind[d] == "drop" for _, _, _, d in o1)
