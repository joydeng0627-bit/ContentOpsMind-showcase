"""Worker-local model resource control. Async resources are created at runtime."""

import asyncio
import functools
import os
import random
import time
import weakref
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from prometheus_client import Counter, Gauge, Histogram
from core.circuit_breaker import CircuitBreaker

IN_FLIGHT = Gauge("contentops_llm_in_flight", "Model attempts in flight")
QUEUED = Gauge("contentops_llm_queued", "Calls waiting for a permit")
WAIT = Histogram("contentops_llm_wait_seconds", "Permit wait time")
LATENCY = Histogram("contentops_llm_attempt_seconds", "Model attempt duration")
RESULTS = Counter("contentops_llm_calls_total", "Logical call outcomes", ["outcome"])
ATTEMPTS = Counter(
    "contentops_llm_attempts_total", "Transport attempt outcomes", ["outcome"]
)
RETRIES = Counter("contentops_llm_retries_total", "Transport retries")


class LLMError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass
class TaskBudget:
    remaining: int
    cancelled: bool = False


_BUDGET = ContextVar("llm_attempt_budget", default=None)
_DEADLINE = ContextVar("llm_deadline", default=None)
_RUNTIME = ContextVar("llm_runtime", default=None)
_RUN_TIMES = weakref.WeakKeyDictionary()


def ensure_task_active():
    budget = _BUDGET.get()
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if (budget and budget.cancelled) or (task and task.cancelling()):
        raise asyncio.CancelledError()
    deadline = _DEADLINE.get()
    if deadline and time.monotonic() >= deadline:
        raise LLMError("llm_deadline_exceeded")


def remaining_timeout(limit):
    ensure_task_active()
    deadline = _DEADLINE.get()
    return min(limit, deadline - time.monotonic()) if deadline else limit


def llm_task(
    seconds_env="LLM_TASK_TIMEOUT_SECONDS",
    seconds=300,
    attempts_env="LLM_TASK_ATTEMPT_BUDGET",
    attempts=64,
):
    def decorate(fn):
        @functools.wraps(fn)
        async def wrapped(*args, **kwargs):
            default_seconds = seconds(*args, **kwargs) if callable(seconds) else seconds
            duration = (
                float(os.getenv(seconds_env, str(default_seconds)))
                if seconds_env
                else float(default_seconds)
            )
            limit = int(os.getenv(attempts_env, str(attempts)))
            if duration <= 0 or limit < 1:
                raise ValueError("Invalid task resource configuration")
            parent = _DEADLINE.get()
            deadline = (
                min(parent, time.monotonic() + duration)
                if parent
                else time.monotonic() + duration
            )
            owns_budget = _BUDGET.get() is None
            budget = _BUDGET.get() or TaskBudget(limit)
            dt, bt = _DEADLINE.set(deadline), _BUDGET.set(budget)
            try:
                async with asyncio.timeout_at(deadline):
                    return await fn(*args, **kwargs)
            except asyncio.CancelledError:
                budget.cancelled = True
                raise
            except TimeoutError:
                if owns_budget:
                    budget.cancelled = True
                raise LLMError("llm_deadline_exceeded") from None
            finally:
                _DEADLINE.reset(dt)
                _BUDGET.reset(bt)

        return wrapped

    return decorate


def _number(name, default, cast=float):
    result = cast(os.getenv(name, str(default)))
    if result <= 0:
        raise ValueError("Invalid " + name)
    return result


class LLMRuntime:
    def __init__(
        self,
        concurrency=None,
        queue_timeout=None,
        call_timeout=None,
        max_attempts=None,
        failure_threshold=5,
        recovery_s=60,
        break_on_429=None,
        backoff=0.5,
    ):
        self.loop = asyncio.get_running_loop()
        concurrency = (
            concurrency
            if concurrency is not None
            else _number("LLM_MAX_IN_FLIGHT", 8, int)
        )
        self.semaphore = asyncio.Semaphore(concurrency)
        self.queue_timeout = (
            queue_timeout
            if queue_timeout is not None
            else _number("LLM_QUEUE_TIMEOUT_SECONDS", 10)
        )
        self.call_timeout = (
            call_timeout
            if call_timeout is not None
            else _number("LLM_CALL_TIMEOUT_SECONDS", 60)
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else _number("LLM_MAX_ATTEMPTS", 2, int)
        )
        if (
            min(
                concurrency,
                self.queue_timeout,
                self.call_timeout,
                self.max_attempts,
                failure_threshold,
            )
            <= 0
            or recovery_s < 0
            or backoff < 0
        ):
            raise ValueError("Invalid LLM resource configuration")
        policy = os.getenv("LLM_BREAK_ON_429", "false").lower()
        if policy not in {"true", "false"}:
            raise ValueError("LLM_BREAK_ON_429 must be true or false")
        self.break_on_429 = policy == "true" if break_on_429 is None else break_on_429
        self.threshold, self.recovery_s, self.backoff = (
            failure_threshold,
            recovery_s,
            backoff,
        )
        self.clients, self.breakers = {}, {}
        self.pending = set()
        self.closed = False

    async def close(self):
        self.closed = True
        for task in list(self.pending):
            task.cancel()
        if self.pending:
            await asyncio.wait(list(self.pending), timeout=1)
        for client in self.clients.values():
            await client.close()
        self.clients.clear()

    async def call(self, client, **kwargs):
        if self.closed or asyncio.get_running_loop() is not self.loop:
            raise RuntimeError("LLM runtime closed or used across event loops")
        ensure_task_active()
        if isinstance(client, SharedClient):
            if client.loop is not None and client.loop is not self.loop:
                raise RuntimeError("Shared model client used across event loops")
            client.loop = self.loop
            identity = client.identity
            if identity not in self.clients:
                self.clients[identity] = client.factory(**client.kwargs, max_retries=0)
            raw = self.clients[identity]
        else:
            identity, raw = ("injected",), client
        # Static service labels; credentials/URLs never become metric labels.
        if identity not in self.breakers:
            self.breakers[identity] = CircuitBreaker(
                self.threshold, self.recovery_s, kind="llm", service="configured"
            )
        breaker = self.breakers[identity]
        ticket = breaker.acquire()
        if ticket is None:
            RESULTS.labels("circuit_open").inc()
            raise LLMError("llm_circuit_open")
        outcome = "neutral"
        try:
            for attempt in range(self.max_attempts):
                deadline = _DEADLINE.get() or (
                    time.monotonic() + self.call_timeout + self.queue_timeout
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMError("llm_deadline_exceeded")
                budget = _BUDGET.get()
                if budget and budget.remaining <= 0:
                    raise LLMError("llm_attempt_budget_exhausted")
                start = time.monotonic()
                QUEUED.inc()
                try:
                    await asyncio.wait_for(
                        self.semaphore.acquire(), min(self.queue_timeout, remaining)
                    )
                except TimeoutError:
                    raise LLMError(
                        "llm_deadline_exceeded"
                        if time.monotonic() >= deadline
                        else "llm_queue_timeout"
                    ) from None
                finally:
                    QUEUED.dec()
                    WAIT.observe(time.monotonic() - start)
                error = None
                model_task = None
                IN_FLIGHT.inc()
                start = time.monotonic()
                try:
                    ensure_task_active()
                    if not breaker.current(ticket):
                        raise LLMError("llm_circuit_open")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LLMError("llm_deadline_exceeded")
                    if budget:
                        if budget.remaining <= 0:
                            raise LLMError("llm_attempt_budget_exhausted")
                        budget.remaining -= 1
                    try:
                        model_task = asyncio.ensure_future(
                            raw.messages.create(**kwargs)
                        )
                        self.pending.add(model_task)

                        def release(completed):
                            self.pending.discard(completed)
                            IN_FLIGHT.dec()
                            self.semaphore.release()
                            if not completed.cancelled():
                                completed.exception()

                        model_task.add_done_callback(release)
                        done, _ = await asyncio.wait(
                            {model_task}, timeout=min(self.call_timeout, remaining)
                        )
                        if not done:
                            model_task.cancel()
                            raise TimeoutError()
                        response = model_task.result()
                    except TimeoutError:
                        error = "timeout"
                    except Exception as ex:
                        status = getattr(ex, "status_code", None)
                        error = (
                            "rate_limited"
                            if status == 429
                            else (
                                "server_error"
                                if status and status >= 500
                                else (
                                    "transport"
                                    if isinstance(ex, (ConnectionError, OSError))
                                    or type(ex).__name__
                                    in {"APIConnectionError", "APITimeoutError"}
                                    else "client_error"
                                )
                            )
                        )
                    if error is None:
                        ensure_task_active()
                        ATTEMPTS.labels("success").inc()
                        outcome = "success"
                        RESULTS.labels("success").inc()
                        return response
                    ATTEMPTS.labels(error).inc()
                finally:
                    LATENCY.observe(time.monotonic() - start)
                    if model_task is None:
                        IN_FLIGHT.dec()
                        self.semaphore.release()
                    elif not model_task.done():
                        model_task.cancel()
                outcome = (
                    "failure"
                    if error in {"timeout", "transport", "server_error"}
                    or (error == "rate_limited" and self.break_on_429)
                    else "neutral"
                )
                if (
                    error == "client_error"
                    or attempt + 1 >= self.max_attempts
                    or ticket.probe
                ):
                    raise LLMError(
                        {
                            "timeout": "llm_call_timeout",
                            "transport": "llm_transport_error",
                            "rate_limited": "llm_rate_limited",
                            "client_error": "llm_client_error",
                            "server_error": "llm_service_error",
                        }[error]
                    )
                if budget and budget.remaining <= 0:
                    raise LLMError("llm_attempt_budget_exhausted")
                delay = random.uniform(0, min(4, self.backoff * 2**attempt))
                if time.monotonic() + delay >= deadline:
                    raise LLMError("llm_deadline_exceeded")
                RETRIES.inc()
                await asyncio.sleep(delay)
        except asyncio.CancelledError:
            RESULTS.labels("cancelled").inc()
            outcome = "neutral"
            raise
        except LLMError as ex:
            if ex.code in {
                "llm_queue_timeout",
                "llm_deadline_exceeded",
                "llm_attempt_budget_exhausted",
                "llm_circuit_open",
            }:
                outcome = "neutral"
            RESULTS.labels(ex.code).inc()
            raise
        finally:
            breaker.finish(ticket, outcome)


class SharedClient:
    def __init__(self, factory, kwargs):
        self.factory, self.kwargs = factory, kwargs
        self.identity = (factory, tuple(sorted(kwargs.items())))
        self.loop = None


def shared_client(factory, **kwargs):
    return SharedClient(factory, kwargs)


def get_runtime():
    runtime = _RUNTIME.get()
    if runtime is not None:
        return runtime
    loop = asyncio.get_running_loop()
    runtime = _RUN_TIMES.get(loop)
    if runtime is None:
        runtime = LLMRuntime()
        _RUN_TIMES[loop] = runtime
    return runtime


async def gateway_call(client, **kwargs):
    if _BUDGET.get() is None:
        return await llm_task()(get_runtime().call)(client, **kwargs)
    return await get_runtime().call(client, **kwargs)


async def close_runtime():
    loop = asyncio.get_running_loop()
    runtime = _RUN_TIMES.pop(loop, None)
    if runtime:
        await runtime.close()


def managed_lifespan(fn):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        get_runtime()
        generator = fn(*args, **kwargs)
        try:
            async for value in generator:
                yield value
        finally:
            try:
                await generator.aclose()
            finally:
                await close_runtime()

    return wrapped


@asynccontextmanager
async def runtime_scope(runtime=None):
    """Explicit ownership for tests and standalone async entrypoints."""
    runtime = runtime or LLMRuntime()
    token = _RUNTIME.set(runtime)
    try:
        yield runtime
    finally:
        _RUNTIME.reset(token)
        await runtime.close()


def managed_entrypoint(fn):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        from dotenv import load_dotenv

        load_dotenv()
        async with runtime_scope():
            return await fn(*args, **kwargs)

    return wrapped
