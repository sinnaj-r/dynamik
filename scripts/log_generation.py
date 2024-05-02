from datetime import datetime, timezone

from prosimos.simulation_engine import run_simulation

bpmn_model = "../data/models/base.model.bpmn"
base_scenario = "../data/scenarios/base.scenario.json"
alternative_scenarios = (
    "batching",
    "contention1",
    "contention2",
    "prioritization",
    "unavailability1",
    "unavailability2",
)
total_cases = 10_000
base_start_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
alternative_start_date = datetime(2020, 3, 10, 12, 0, 0, tzinfo=timezone.utc)

print("Generating base log...")
run_simulation(
    starting_at=str(base_start_date),
    total_cases=total_cases,
    bpmn_path=bpmn_model,
    json_path=base_scenario,
    log_out_path="../data/logs/base.log.csv",
    stat_out_path="../data/stats/base.stat.csv",
)

# change case identifiers so they are easily recognizable
with open(f"../data/logs/base.log.csv", 'r+') as file:
    lines = file.readlines()
    header = lines[0]
    body = [f"base-{line}" for line in lines[1:]]

    file.seek(0)
    file.writelines(header)
    file.writelines(body)
    file.truncate()

for alternative_scenario in alternative_scenarios:
    print(f"Generating {alternative_scenario} log...")

    run_simulation(
        starting_at=str(alternative_start_date),
        total_cases=total_cases,
        bpmn_path=bpmn_model,
        json_path=f"../data/scenarios/{alternative_scenario}.scenario.json",
        log_out_path=f"../data/logs/{alternative_scenario}.log.csv",
        stat_out_path=f"../data/stats/{alternative_scenario}.stat.csv",
    )

    # change case identifiers so they are easily recognizable
    with open(f"../data/logs/{alternative_scenario}.log.csv", 'r+') as file:
        lines = file.readlines()
        header = lines[0]
        body = [f"{alternative_scenario}-{line}" for line in lines[1:]]

        file.seek(0)
        file.writelines(header)
        file.writelines(body)
        file.truncate()
