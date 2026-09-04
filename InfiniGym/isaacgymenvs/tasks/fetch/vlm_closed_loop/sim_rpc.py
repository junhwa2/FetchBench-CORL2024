"""File-based local RPC for the VLM+VORM closed loop (see VLM_SIM_RPC_SPEC.md).

Pure-Python, no isaacgym / torch dependency, so it can be imported from either
conda env and unit-tested on its own. Implements the single-in-flight
request/response protocol:

    req/<id>.json (+ req/<id>.ready)   written by the CLIENT (vorm master)
    resp/<id>.json(+ resp/<id>.ready)  written by the SERVER (FetchBench sim)
    payload/*                          large binaries referenced by relative path

Two rules only:
  1. JSON is published atomically: write `<x>.tmp`, then os.replace -> `<x>`.
  2. `.ready` is the sole barrier: the producer writes every payload, then the
     JSON, then finally touches `.ready`; the consumer waits only on `.ready`.

`RESP_TIMEOUT`/`POLL_SEC` are advisory defaults; per-call overrides are allowed.
"""

import json
import os
import time

PROTOCOL_VERSION = 0
POLL_SEC = 0.02
RESP_TIMEOUT = 600.0  # seconds; None => wait forever


# --------------------------------------------------------------------------- #
# low-level file helpers
# --------------------------------------------------------------------------- #
def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        # default=str so a stray numpy array / tensor in a response body can
        # never crash the serve loop (it stringifies instead). The JSON encode
        # happens OUTSIDE the handler try, so without this a non-serializable
        # field would kill the server and hang the client on its resp barrier.
        json.dump(obj, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _touch(path):
    with open(path, "w") as f:
        f.flush()
        os.fsync(f.fileno())


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _wait_for(path, timeout, poll):
    """Block until `path` exists. timeout=None waits forever. Returns True on
    success, raises TimeoutError otherwise."""
    t0 = time.time()
    while not os.path.exists(path):
        if timeout is not None and (time.time() - t0) > timeout:
            raise TimeoutError("timed out waiting for {}".format(path))
        time.sleep(poll)
    return True


# --------------------------------------------------------------------------- #
# directory layout
# --------------------------------------------------------------------------- #
class _Layout(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.req = os.path.join(self.root, "req")
        self.resp = os.path.join(self.root, "resp")
        self.payload = os.path.join(self.root, "payload")

    def ensure(self):
        for d in (self.req, self.resp, self.payload):
            os.makedirs(d, exist_ok=True)

    def req_json(self, i):
        return os.path.join(self.req, "{}.json".format(i))

    def req_ready(self, i):
        return os.path.join(self.req, "{}.ready".format(i))

    def resp_json(self, i):
        return os.path.join(self.resp, "{}.json".format(i))

    def resp_ready(self, i):
        return os.path.join(self.resp, "{}.ready".format(i))

    def payload_path(self, name):
        return os.path.join(self.payload, name)

    def rel(self, abs_path):
        """Path relative to root, using forward slashes (spec: payload refs are
        relative to $VLM_RPC_DIR)."""
        return os.path.relpath(abs_path, self.root).replace(os.sep, "/")

    def abs(self, rel_path):
        return os.path.join(self.root, rel_path)


# --------------------------------------------------------------------------- #
# server (FetchBench sim side). The matching CLIENT half lives on the vorm
# master in vorm_pipeline/sim_backend_mj.py (kept separate so the two conda
# envs stay independent); this module is server-only.
# --------------------------------------------------------------------------- #
class RpcServer(object):
    """Serves requests by calling `handler(method, req) -> dict`.

    The handler returns the response body (without the ok/v/id envelope). Any
    exception is caught and reported as ok=False with a traceback so the client
    fails loudly instead of hanging. Loop exits after replying to `close`.
    """

    def __init__(self, root, handler, poll=POLL_SEC):
        self.io = _Layout(root)
        self.io.ensure()
        self.handler = handler
        self.poll = poll

    def serve_forever(self):
        import traceback as _tb
        next_id = 0
        while True:
            _wait_for(self.io.req_ready(next_id), timeout=None, poll=self.poll)
            req = _read_json(self.io.req_json(next_id))
            method = req.get("method", "")
            try:
                body = self.handler(method, req) or {}
                resp = {"v": PROTOCOL_VERSION, "id": next_id, "ok": True}
                resp.update(body)
            except Exception as exc:  # noqa: BLE001 - surfaced to the client
                resp = {"v": PROTOCOL_VERSION, "id": next_id, "ok": False,
                        "error": "{}: {}".format(type(exc).__name__, exc),
                        "traceback": _tb.format_exc()}
            _atomic_write_json(self.io.resp_json(next_id), resp)
            _touch(self.io.resp_ready(next_id))
            if method == "close":
                break
            next_id += 1
