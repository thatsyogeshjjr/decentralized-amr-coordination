from fleet.launch import parse_overrides


def test_override_parsing_types_and_strings():
    o = parse_overrides(["comms.loss=0.1", "fleet.learned_priority=true", "run.name=fourway__pibt",
                         "world.map=configs/maps/topo_fourway.txt", "tasks.list=[[0, [4, 1], [4, 11]]]",
                         "run.name2=false_start", "faults.random_blockages.rate=0.2"])
    assert o["comms"]["loss"] == 0.1 and o["fleet"]["learned_priority"] is True
    assert o["run"]["name"] == "fourway__pibt"          # regression: names starting with 'f'/'t' stay strings
    assert o["run"]["name2"] == "false_start"
    assert o["world"]["map"] == "configs/maps/topo_fourway.txt"
    assert o["tasks"]["list"] == [[0, [4, 1], [4, 11]]]
    assert o["faults"]["random_blockages"]["rate"] == 0.2
