from epd.input.mapping import Mapping

DEFAULT_CSV_MAPPING: Mapping = Mapping(start="start", end="end", case="case", activity="activity", resource="resource")

DEFAULT_APROMORE_CSV_MAPPING: Mapping = Mapping(start="start_time", end="end_time", case="case_id",
                                                activity="Activity", resource="Resource")
