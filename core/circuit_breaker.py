"""Small synchronous state machine; no network work occurs under its lock."""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from prometheus_client import Counter, Gauge

STATE = Gauge(
    "contentops_circuit_state", "0 closed, 1 open, 2 half open", ["kind", "service"]
)
TRANSITIONS = Counter(
    "contentops_circuit_transitions_total",
    "Circuit transitions",
    ["kind", "service", "state"],
)
PROBES = Counter(
    "contentops_circuit_probes_total", "Recovery probes", ["kind", "service", "outcome"]
)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class Ticket:
    generation: int
    probe: bool


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold=5,
        recovery_s=60.0,
        probe_limit=1,
        kind="mcp",
        service="unregistered",
        clock=time.monotonic,
    ):
        self.threshold, self.recovery_s = failure_threshold, recovery_s
        self.probe_limit, self.kind, self.service = probe_limit, kind, service
        self.state, self.fail_count, self.opened_at = CircuitState.CLOSED, 0, None
        self.generation, self.probes = 0, 0
        self._clock, self._lock = clock, threading.Lock()

    def _transition(self, state):
        self.state = state
        self.generation += 1
        self.probes = 0
        if state == CircuitState.OPEN:
            self.opened_at = self._clock()
        STATE.labels(self.kind, self.service).set(
            {CircuitState.CLOSED: 0, CircuitState.OPEN: 1, CircuitState.HALF_OPEN: 2}[
                state
            ]
        )
        TRANSITIONS.labels(self.kind, self.service, state.value).inc()

    def acquire(self):
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._clock() - self.opened_at < self.recovery_s:
                    return None
                self._transition(CircuitState.HALF_OPEN)
            probe = self.state == CircuitState.HALF_OPEN
            if probe:
                if self.probes >= self.probe_limit:
                    return None
                self.probes += 1
            return Ticket(self.generation, probe)

    def finish(self, ticket, outcome):
        with self._lock:
            if ticket.generation != self.generation:
                return
            if ticket.probe:
                PROBES.labels(self.kind, self.service, outcome).inc()
                self.fail_count = 0 if outcome == "success" else self.threshold
                self._transition(
                    CircuitState.CLOSED if outcome == "success" else CircuitState.OPEN
                )
            elif outcome == "success":
                self.fail_count = 0
            elif outcome == "failure":
                self.fail_count += 1
                if self.fail_count >= self.threshold:
                    self._transition(CircuitState.OPEN)

    def current(self, ticket):
        with self._lock:
            return (
                ticket.generation == self.generation and self.state != CircuitState.OPEN
            )
