"""Warm fork-server execution backend for the DS-1000 potential.

One worker per (python_executable, extra_env, event loop) pre-imports the
science libraries and forks an isolated child per harness script, cutting
per-check cost from seconds to milliseconds. Any backend failure raises
ForkserverUnavailable; callers fall back to the plain subprocess path.
"""

import asyncio
import atexit
import json
import os
import signal
import tempfile

_WORKER_PATH = os.path.join(os.path.dirname(__file__), "_forkserver_worker.py")
_STREAM_LIMIT = 32 * 1024 * 1024
_MAX_START_ATTEMPTS = 3


class ForkserverUnavailable(RuntimeError):
    """The fork-server cannot serve requests; use the subprocess fallback."""


class ForkserverExecutor:
    def __init__(self, python_executable: str, extra_env=None):
        self.python_executable = python_executable
        self.extra_env = dict(extra_env or {})
        self.proc = None
        self._futures = {}
        self._next_id = 0
        self._wlock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._start_attempts = 0
        self.failed = False
        self.loop = None
        self.stderr_log = None

    async def _ensure_started(self):
        async with self._start_lock:
            if self.failed:
                raise ForkserverUnavailable("fork-server permanently failed")
            if self.proc is not None and self.proc.returncode is None:
                return
            if self._start_attempts >= _MAX_START_ATTEMPTS:
                self.failed = True
                raise ForkserverUnavailable("fork-server start attempts exhausted")
            self._start_attempts += 1
            try:
                env = dict(os.environ)
                env.update(self.extra_env)
                log = tempfile.NamedTemporaryFile(
                    mode="w", prefix="ds1000_worker_", suffix=".log", delete=False
                )
                self.stderr_log = log.name
                self.proc = await asyncio.create_subprocess_exec(
                    self.python_executable,
                    "-u",
                    _WORKER_PATH,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=log,
                    env=env,
                    limit=_STREAM_LIMIT,
                )
                ready = json.loads(
                    await asyncio.wait_for(self.proc.stdout.readline(), timeout=300)
                )
                if not ready.get("ready"):
                    raise RuntimeError(f"unexpected worker handshake: {ready!r}")
            except Exception as exc:
                self.kill()
                raise ForkserverUnavailable(f"fork-server start failed: {exc}") from exc
            # Fresh futures per worker generation: a late cleanup from a dead
            # generation's reader must not clobber the new generation.
            self._futures = {}
            asyncio.get_running_loop().create_task(
                self._reader(self.proc, self._futures)
            )

    async def _reader(self, proc, futures):
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)
                fut = futures.pop(msg["id"], None)
                if fut is not None and not fut.done():
                    fut.set_result(msg["out"])
        except Exception:  # noqa: BLE001
            pass
        finally:
            for fut in futures.values():
                if not fut.done():
                    fut.set_exception(ForkserverUnavailable("worker stream ended"))
            futures.clear()

    async def run(self, script: str, timeout: float):
        """
        Run a harness script in a forked child. Returns combined stdout+stderr,
        or None on timeout/child crash. Raises ForkserverUnavailable on backend
        failure (caller falls back to a plain subprocess).
        """
        return await self._request(
            {"script": script, "timeout": timeout},
            timeout,
            wait_extra=30,
            congestion_raises=True,
        )

    async def run_session(self, skey, setup, body, fallback, timeout):
        """
        Run `body` in a warm per-task session (setup executed once per skey);
        the worker runs `fallback` through the plain path if the session is
        unusable. Same return contract as run().
        """
        # Session requests may wait behind one-off setup (<=120s) and the
        # per-task serialized checks, so budget slack from the timeout.
        return await self._request(
            {
                "skey": skey,
                "setup": setup,
                "body": body,
                "fallback": fallback,
                "timeout": timeout,
            },
            timeout,
            wait_extra=120 + 10 * timeout,
        )

    async def _request(
        self,
        payload_fields: dict,
        timeout: float,
        wait_extra: float = 30,
        congestion_raises: bool = False,
    ):
        await self._ensure_started()
        self._next_id += 1
        rid = self._next_id
        fut = asyncio.get_running_loop().create_future()
        self._futures[rid] = fut
        payload = json.dumps({"id": rid, **payload_fields}) + "\n"
        try:
            async with self._wlock:
                self.proc.stdin.write(payload.encode())
                await self.proc.stdin.drain()
        except Exception as exc:
            self._futures.pop(rid, None)
            raise ForkserverUnavailable(f"fork-server write failed: {exc}") from exc
        try:
            out = await asyncio.wait_for(fut, timeout=timeout + wait_extra)
        except asyncio.TimeoutError:
            self._futures.pop(rid, None)
            # No worker verdict at all: backend congestion, not a script
            # timeout (the worker stamps those). Strict callers re-run via
            # the subprocess fallback rather than misreading it as -inf.
            if congestion_raises:
                raise ForkserverUnavailable("fork-server response overdue")
            return None
        # The worker reports timeout/crash via an id-stamped final line that
        # solution prints cannot spoof.
        last = out.rstrip().rsplit("\n", 1)[-1]
        if last.startswith(f"<<<WORKER {rid} "):
            return None
        return out

    def kill(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.returncode is None:
            # os.kill is independent of the (possibly closed) event loop.
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass


_executors = {}


def shared_executor(python_executable: str, extra_env=None) -> ForkserverExecutor:
    """Executor shared per (python, extra_env, running event loop)."""
    loop = asyncio.get_running_loop()
    for k, ex in list(_executors.items()):
        if ex.loop is not None and ex.loop.is_closed():
            ex.kill()
            del _executors[k]
    key = (id(loop), python_executable, frozenset((extra_env or {}).items()))
    executor = _executors.get(key)
    if executor is None:
        executor = ForkserverExecutor(python_executable, extra_env)
        executor.loop = loop
        _executors[key] = executor
    return executor


@atexit.register
def _cleanup():
    for executor in _executors.values():
        executor.kill()
