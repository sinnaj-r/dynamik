from epd.model import Event


def always_true(_: Event) -> bool:
    return True


def always_false(_: Event) -> bool:
    return False


def has_resource(evt: Event) -> bool:
    return evt.resource is not None


def has_duration(evt: Event) -> bool:
    return (evt.end - evt.start).total_seconds() > 0
