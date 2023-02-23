from dataclasses import dataclass


@dataclass
class EventFields:
    START: str
    END: str
    CASE: str
    ACTIVITY: str
    RESOURCE: str


DEFAULT_EVENT_FIELDS = EventFields(
    START='start_time',
    END='end_time',
    CASE='case_id',
    ACTIVITY='Activity',
    RESOURCE='Resource'
)
