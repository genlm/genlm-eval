#!/usr/bin/env python
"""
Warm fork-server for DS-1000 harness scripts: pre-imports the science libs
(tensorflow excluded, fork-unsafe), forks an isolated child per request.
Session requests keep a per-task child alive past setup and fork checks from
it, serialized per task; session failures run the request's fallback script.

JSON lines, stdin -> stdout {"id", "out"}; {"ready": true} after warmup.
plain: {"id", "script", "timeout"}; session: {"id", "skey", "setup", "body",
"fallback", "timeout"}. Worker timeout/crash verdicts end with an id-stamped
"<<<WORKER <id> ...>>>" line.
"""

import json
import os
import time
import select
import shutil
import signal
import sys
import tempfile
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Pin BLAS/OpenMP pools to one thread before importing: children inherit only
# the forking thread, so multi-thread pools can deadlock post-fork.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

_PRELOAD = os.environ.get(
    "DS1000_FORKSERVER_PRELOAD",
    "numpy,pandas,scipy,matplotlib,matplotlib.pyplot,PIL.Image,seaborn,"
    "sklearn,sklearn.preprocessing,torch",
)
for _mod in filter(None, _PRELOAD.split(",")):
    try:
        __import__(_mod)
    except Exception:  # noqa: BLE001
        pass

try:
    import torch

    torch.set_num_threads(1)
except Exception:  # noqa: BLE001
    pass

# Backstop against fork-bombing if a client ever pipelines more requests than
# its own concurrency gate allows.
MAX_CHILDREN = int(os.environ.get("DS1000_FORKSERVER_MAX_CHILDREN", "64"))
MAX_SESSIONS = int(os.environ.get("DS1000_FORKSERVER_MAX_SESSIONS", "64"))
SETUP_TIMEOUT = int(os.environ.get("DS1000_FORKSERVER_SETUP_TIMEOUT", "120"))


def _sandbox_into(td):
    os.chdir(td)
    os.environ.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": td,
            "XDG_CACHE_HOME": os.path.join(td, "xdg_cache"),
            "XDG_CONFIG_HOME": os.path.join(td, "xdg_config"),
            "PYTHONWARNINGS": "ignore",
        }
    )


def run_child(script: str, timeout: float, out_path: str, td: str) -> None:
    """Executed in the forked child. Never returns."""
    rc = 0
    try:
        fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        _sandbox_into(td)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(max(1, int(timeout)))
        g = {"__name__": "__main__", "__builtins__": __builtins__}
        try:
            exec(compile(script, "<harness>", "exec"), g, g)
        except SystemExit:
            pass
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            rc = 1
    except BaseException:  # noqa: BLE001
        rc = 2
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(rc)


def run_session(setup: str, cmd_r: int, res_w: int, td: str) -> None:
    """Session child: run setup once, then fork one grandchild per body."""

    def send(obj):
        os.write(res_w, (json.dumps(obj) + "\n").encode())

    try:
        os.setsid()
        log_fd = os.open(os.path.join(td, "__session__.log"),
                         os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        os.close(log_fd)
        _sandbox_into(td)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(SETUP_TIMEOUT)
        ns = {"__name__": "__main__", "__builtins__": __builtins__}
        try:
            exec(compile(setup, "<session-setup>", "exec"), ns, ns)
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            send({"setup_failed": 1})
            os._exit(0)
        signal.alarm(0)
        send({"ready": 1})

        buf = b""
        while True:
            chunk = os.read(cmd_r, 1 << 20)
            if not chunk:
                os._exit(0)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                req = json.loads(line)
                sys.stdout.flush()
                sys.stderr.flush()
                try:
                    pid = os.fork()
                except OSError:
                    send({"id": req["id"], "forkfail": 1})
                    continue
                if pid == 0:
                    os.close(cmd_r)
                    os.close(res_w)
                    run_child_in_ns(req["body"], req["timeout"], req["out"], ns)
                _, status = os.waitpid(pid, 0)
                send({"id": req["id"], "status": status})
    except BaseException:  # noqa: BLE001
        try:
            send({"setup_failed": 1})
        except Exception:  # noqa: BLE001
            pass
        os._exit(2)


def run_child_in_ns(body: str, timeout: float, out_path: str, ns) -> None:
    """Grandchild: execute a body script in the session namespace."""
    rc = 0
    try:
        fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        os.close(fd)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.alarm(max(1, int(timeout)))
        try:
            exec(compile(body, "<harness-body>", "exec"), ns, ns)
        except SystemExit:
            pass
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            rc = 1
    except BaseException:  # noqa: BLE001
        rc = 2
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        os._exit(rc)


def _read_out(out_path):
    try:
        with open(out_path, "r", errors="replace") as fh:
            out = fh.read()
    except OSError:
        out = ""
    # Solutions may print huge arrays; markers are emitted at the end, so
    # keep the head for context and the full tail.
    if len(out) > 100_000:
        out = out[:8_000] + "\n<<<TRUNCATED>>>\n" + out[-64_000:]
    return out


def _suffix_for_status(status, req_id):
    """Worker verdict suffix, id-stamped so solution prints cannot spoof it."""
    if os.WIFSIGNALED(status) and os.WTERMSIG(status) == signal.SIGALRM:
        return f"\n<<<WORKER {req_id} TIMEOUT>>>"
    if os.WIFSIGNALED(status):
        return f"\n<<<WORKER {req_id} CHILD_EXIT -{os.WTERMSIG(status)}>>>"
    if os.WEXITSTATUS(status) == 2:
        return f"\n<<<WORKER {req_id} CHILD_EXIT 2>>>"
    return ""


def main() -> None:
    children = {}  # pid -> (req_id, out_path, td, deadline)
    sessions = {}  # skey -> session dict
    session_pids = {}  # pid -> skey

    def respond(req_id, out):
        sys.stdout.write(json.dumps({"id": req_id, "out": out}) + "\n")
        sys.stdout.flush()

    def spawn_plain(req_id, script, timeout):
        td = tempfile.mkdtemp(prefix="ds1000_fork_")
        out_path = os.path.join(td, "__out__.txt")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            pid = os.fork()
        except OSError:
            shutil.rmtree(td, ignore_errors=True)
            respond(req_id, f"\n<<<WORKER {req_id} CHILD_EXIT fork>>>")
            return
        if pid == 0:
            run_child(script, timeout, out_path, td)
        children[pid] = (req_id, out_path, td, time.time() + timeout + 15)

    def run_fallback(req):
        spawn_plain(req["id"], req["fallback"], req.get("timeout", 30))

    def kill_session(skey):
        s = sessions.pop(skey, None)
        if s is None:
            return
        session_pids.pop(s["pid"], None)
        for fd in (s["cmd_w"], s["res_r"]):
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.killpg(s["pid"], signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        shutil.rmtree(s["td"], ignore_errors=True)

    def fail_session(skey):
        """Session became unusable: serve queued/inflight via fallback."""
        s = sessions.get(skey)
        if s is None:
            return
        pending = list(s["queue"])
        if s["inflight"] is not None:
            pending.insert(0, s["inflight"])
            s["inflight"] = None
        s["queue"].clear()
        kill_session(skey)
        sessions[skey] = {"state": "bad"}
        for req in pending:
            run_fallback(req)

    def spawn_session(skey, setup):
        cmd_r, cmd_w = os.pipe()
        res_r, res_w = os.pipe()
        td = tempfile.mkdtemp(prefix="ds1000_sess_")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            pid = os.fork()
        except OSError:
            shutil.rmtree(td, ignore_errors=True)
            for fd in (cmd_r, cmd_w, res_r, res_w):
                os.close(fd)
            sessions[skey] = {"state": "bad"}
            return
        if pid == 0:
            os.close(cmd_w)
            os.close(res_r)
            run_session(setup, cmd_r, res_w, td)
        os.close(cmd_r)
        os.close(res_w)
        sessions[skey] = {
            "state": "starting", "pid": pid, "cmd_w": cmd_w, "res_r": res_r,
            "td": td, "queue": [], "inflight": None, "buf": b"", "seq": 0,
        }
        session_pids[pid] = skey

    def pump(skey):
        s = sessions.get(skey)
        if not s or s.get("state") != "ready" or s["inflight"] or not s["queue"]:
            return
        req = s["queue"].pop(0)
        s["inflight"] = req
        req["_deadline"] = time.time() + req.get("timeout", 30) + 15
        s["seq"] += 1
        out_path = os.path.join(s["td"], f"__out_{s['seq']}__.txt")
        req["_out_path"] = out_path
        msg = {"id": req["id"], "body": req["body"],
               "timeout": req.get("timeout", 30), "out": out_path}
        try:
            os.write(s["cmd_w"], (json.dumps(msg) + "\n").encode())
        except OSError:
            fail_session(skey)

    def on_session_msg(skey, msg):
        s = sessions.get(skey)
        if s is None or "pid" not in s:
            return
        if msg.get("ready"):
            s["state"] = "ready"
            pump(skey)
        elif msg.get("setup_failed"):
            fail_session(skey)
        elif "id" in msg:
            req = s["inflight"]
            s["inflight"] = None
            if req is not None and req["id"] != msg["id"]:
                run_fallback(req)
            elif req is not None:
                if msg.get("forkfail"):
                    respond(req["id"], f"\n<<<WORKER {req['id']} CHILD_EXIT fork>>>")
                else:
                    out = _read_out(req["_out_path"])
                    try:
                        os.unlink(req["_out_path"])
                    except OSError:
                        pass
                    respond(req["id"], out + _suffix_for_status(msg["status"], req["id"]))
            pump(skey)

    def handle_request(req):
        if "skey" not in req:
            spawn_plain(req["id"], req["script"], req.get("timeout", 30))
            return
        skey = req["skey"]
        s = sessions.get(skey)
        if s is not None and s.get("state") == "bad":
            run_fallback(req)
            return
        if s is None:
            if len([v for v in sessions.values() if "pid" in v]) >= MAX_SESSIONS:
                evictable = [k for k, v in sessions.items()
                             if v.get("state") == "ready"
                             and not v["inflight"] and not v["queue"]]
                if evictable:
                    kill_session(evictable[0])
                else:
                    run_fallback(req)
                    return
            spawn_session(skey, req["setup"])
            s = sessions.get(skey)
            if s is None or s.get("state") == "bad":
                run_fallback(req)
                return
        s["queue"].append(req)
        pump(skey)

    def enforce_deadlines():
        now = time.time()
        for pid, (_rid, _o, _t, deadline) in list(children.items()):
            if now > deadline:
                # in-child alarm was defeated (or never fired): kill from here;
                # reap() collects and responds with the signal suffix.
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for skey, s in list(sessions.items()):
            req = s.get("inflight") if "pid" in s else None
            if req is not None and now > req.get("_deadline", now + 1):
                fail_session(skey)

    def reap():
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
            if pid in session_pids:
                fail_session(session_pids[pid])
                continue
            if pid not in children:
                continue
            req_id, out_path, td, _deadline = children.pop(pid)
            out = _read_out(out_path) + _suffix_for_status(status, req_id)
            shutil.rmtree(td, ignore_errors=True)
            respond(req_id, out)

    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    buf = b""
    stdin_fd = sys.stdin.fileno()
    eof = False

    def busy():
        return (children
                or any(v.get("inflight") or v.get("queue")
                       for v in sessions.values() if "pid" in v))

    while not eof or busy():
        timeout = 0.05 if busy() else None
        fds = [v["res_r"] for v in sessions.values() if "pid" in v]
        read_stdin = not eof and len(children) < MAX_CHILDREN
        if read_stdin:
            fds.append(stdin_fd)
        ready, _, _ = select.select(fds, [], [], timeout)
        for fd in ready:
            if fd == stdin_fd:
                chunk = os.read(stdin_fd, 1 << 20)
                if not chunk:
                    eof = True
                buf += chunk
                continue
            for skey, s in list(sessions.items()):
                if s.get("res_r") == fd:
                    try:
                        chunk = os.read(fd, 1 << 20)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        fail_session(skey)
                        break
                    s["buf"] += chunk
                    while b"\n" in s["buf"]:
                        line, s["buf"] = s["buf"].split(b"\n", 1)
                        if not line.strip():
                            continue
                        try:
                            msg = json.loads(line)
                        except ValueError:
                            fail_session(skey)
                            break
                        on_session_msg(skey, msg)
                    break
        # Buffered lines wait while the child cap is hit; they imply a live
        # child, so the outer loop cannot exit with work queued.
        while b"\n" in buf and len(children) < MAX_CHILDREN:
            line, buf = buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                req = json.loads(line)
            except ValueError:
                sys.stderr.write("worker: dropping malformed request line\n")
                continue
            handle_request(req)
        enforce_deadlines()
        reap()

    for skey in list(sessions):
        kill_session(skey)
    for pid in list(children):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline_children = dict(children)
    while deadline_children:
        try:
            pid, _ = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        deadline_children.pop(pid, None)


if __name__ == "__main__":
    main()
