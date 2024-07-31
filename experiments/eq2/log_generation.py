from datetime import UTC, datetime

from prosimos.simulation_engine import run_simulation


def _generate_log(
        name: str,
        scenario: str,
        cases: int,
        start_date: datetime,
) -> tuple[datetime, datetime]:
    print(f"Generating {name} log...")

    run_simulation(
        starting_at=str(start_date),
        total_cases=cases,
        bpmn_path="./models/base.model.bpmn",
        json_path=f"./scenarios/{scenario}.scenario.json",
        log_out_path=f"./logs/{name}.log.csv",
        stat_out_path=f"./stats/{name}.stat.csv",
    )

    # get end date
    with open(f"./stats/{name}.stat.csv") as file:
        lines = file.readlines()
        end_date = datetime.fromisoformat(lines[1].split(",")[1].strip())

    # change case identifiers so they are easily recognizable
    with open(f"./logs/{name}.log.csv", 'r+') as file:
        lines = file.readlines()
        header = lines[0]
        body = [f"{name}-{line}" for line in lines[1:]]

        file.seek(0)
        file.writelines(header)
        file.writelines(body)
        file.truncate()

    return start_date, end_date


if __name__ == "__main__":
    total_cases = 10_000
    base_start_date = datetime(2020, 1, 1, 0, 0, 0, tzinfo=UTC)

    # _, scenario10_end = _generate_log("scenario10", "base", total_cases, base_start_date)
    # _, scenario11_end = _generate_log("scenario11", "scenario1", total_cases, scenario10_end)

    # _, scenario20_end = _generate_log("scenario20", "base", total_cases, base_start_date)
    # _, scenario21_end = _generate_log("scenario21", "scenario2", total_cases, scenario20_end)

    # _, scenario30_end = _generate_log("scenario30", "base", total_cases, base_start_date)
    # _, scenario31_end = _generate_log("scenario31", "scenario3", total_cases, scenario30_end)

    # _, scenario40_end = _generate_log("scenario40", "base", total_cases, base_start_date)
    # _, scenario41_end = _generate_log("scenario41", "scenario4", total_cases, scenario40_end)

    _, scenario50_end = _generate_log("scenario50", "base", total_cases, base_start_date)
    _, scenario51_end = _generate_log("scenario51", "scenario5", total_cases, scenario50_end)
