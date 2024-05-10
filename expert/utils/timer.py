from __future__ import annotations

import time
import typing
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps


@dataclass
class Timer:
    """A simple timer for profiling the code"""

    __timers = defaultdict(list)

    def start(self: typing.Self, name: str = "default") -> None:
        """Starts a new timer block with name 'name'"""
        self.__timers[name].append({
            "start": time.perf_counter_ns(),
        })

    def end(self: typing.Self, name: str = "default") -> None:
        """Ends the timer block named 'name'"""
        self.__timers[name][-1]["end"] = time.perf_counter_ns()

    def elapsed(self: typing.Self, name: str = "default") -> timedelta:
        """Returns the elapsed time for the timer with the given name"""
        return sum([timedelta(microseconds=(period["end"] - period["start"]) / 1000) for period in self.__timers[name]], start=timedelta())

    def reset(self: typing.Self, name: str | None = None) -> None:
        if name is None:
            self.__timers = defaultdict(list)
        else:
            self.__timers[name] = []

    @contextmanager
    def profile(self: typing.Self, name: str = "default") -> typing.Generator[Timer, None, None]:
        """Profiles a block of code using a context provider"""
        self.start(name)
        try:
            yield self
        finally:
            self.end(name)

    def __str__(self: typing.Self) -> str:
        sorted_keys = sorted(self.__timers.keys(), key=lambda k: self.elapsed(k), reverse=True)
        return "\n".join([f"{name}(): {self.elapsed(name)}" for name in sorted_keys])


DEFAULT_TIMER = Timer()


def profile(name: str | None = None, *, timer: Timer = DEFAULT_TIMER) -> typing.Callable:
    """Profiles a method or function using a decorator"""
    def decorator(func: typing.Callable) -> typing.Callable:
        @wraps(func)
        def _wrapper(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            with timer.profile(name if name is not None else func.__qualname__):
                return func(*args, **kwargs)
        return _wrapper

    return decorator
