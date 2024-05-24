from datetime import datetime, timezone

from prosimos.simulation_engine import run_simulation


def generate_log(
        scenario: str,
        cases: int,
        start_date: datetime,
):
    print(f"Generating {scenario} log...")

    run_simulation(
        starting_at=str(start_date),
        total_cases=cases,
        bpmn_path="../data/models/base.model.bpmn",
        json_path=f"../data/scenarios/{scenario}.scenario.json",
        log_out_path=f"../data/logs/{scenario}.log.csv",
        stat_out_path=f"../data/stats/{scenario}.stat.csv",
    )

    # change case identifiers so they are easily recognizable
    with open(f"../data/logs/{scenario}.log.csv", 'r+') as file:
        lines = file.readlines()
        header = lines[0]
        body = [f"{scenario}-{line}" for line in lines[1:]]

        file.seek(0)
        file.writelines(header)
        file.writelines(body)
        file.truncate()


if __name__ == "__main__":
    total_cases = 10_000
    base_start_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    alternative_start_date = datetime(2020, 10, 1, 0, 0, 0, tzinfo=timezone.utc)

    generate_log("base", total_cases, base_start_date)
    generate_log("batching", total_cases, alternative_start_date)
    generate_log("contention1", total_cases, alternative_start_date)
    generate_log("contention2", total_cases, alternative_start_date)
    generate_log("prioritization", total_cases, alternative_start_date)
    generate_log("unavailability1", total_cases, alternative_start_date)
    generate_log("unavailability2", total_cases, alternative_start_date)
