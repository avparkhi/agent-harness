"""Step 6: durable sessions. A session survives the process that runs it.

Until now the conversation lived in a Python list. Kill the process and the work is gone,
including any half-finished tool round. Here every step is written to a SQLite file
*before* anything else happens, so a new process can open the file and carry on.

The design follows pi's durable package (earendil-works/pi, packages/durable), cut down
to the core rules:

    1. Commit before you show. Nothing is printed, returned or acted on until it's stored.
    2. A commit is atomic. Several entries written together all land, or none do.
    3. Record intent before effect. A tool call is stored as "started" before it runs, so
       after a crash we know it *may* have happened.
    4. Only replay what's safe. On resume, a started call with no result reruns only if the
       tool is replay-safe (reading a file, writing the same content again). Anything else
       (a shell command, a subagent) is answered with an "interrupted" error, and the model
       decides what to do, the same way it handles any other failed tool call.
    5. Don't crash-loop. Each resume of unfinished work counts; past RESTART_LIMIT the
       session is parked until a person looks at it.
    6. One writer. A lease says which process owns the session. pi assumes one process
       per file; once the work can run on other machines, it's worth writing that down.

The log is append-only. The conversation is never stored as such: it's *derived* from the
entries by fold(), which is also how a resumed process rebuilds it.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field

LEASE_SECONDS = 60
RESTART_LIMIT = 5
# Tools whose effect is the same if run twice. write_file qualifies: writing the same
# content again leaves the same file. run_bash doesn't: `git commit` twice is two commits.
REPLAY_SAFE = {"read_file", "search_files", "write_file", "todo_write"}

SCHEMA = """
create table if not exists sessions (
    id          text primary key,
    created_at  real not null,
    meta        text not null,            -- json: cwd, model, system prompt, sandbox id...
    meter       text not null default '{}',
    owner       text,                     -- host:pid of the process holding the lease
    lease_until real not null default 0,
    restarts    integer not null default 0
);
create table if not exists entries (
    session_id  text not null,
    seq         integer not null,
    kind        text not null,            -- user, assistant, intent, result, todos, compaction, final
    data        text not null,
    at          real not null,
    primary key (session_id, seq)
);
"""


class LeaseHeld(Exception):
    """Another live process owns this session."""


class Parked(Exception):
    """The session crashed too many times in a row; a person should look before it runs again."""


def me() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def _dead_here(owner: str | None) -> bool:
    """True if the owner was a process on this machine that no longer exists. A crash on this
    machine can then be taken over at once instead of waiting for the lease to run out."""
    if not owner or ":" not in owner:
        return True
    host, pid = owner.rsplit(":", 1)
    if host != socket.gethostname():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except (PermissionError, ValueError):
        return False
    return False


class Store:
    """One SQLite file holding any number of sessions."""

    def __init__(self, path):
        # WAL lets readers (`--list`, a viewer) read while the owner writes. synchronous=NORMAL
        # survives a process crash; a power cut can lose the very last commit.
        self.db = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False, timeout=10)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("pragma synchronous=normal")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def tx(self, fn):
        """Run fn(db) in one transaction: all of its writes land, or none do."""
        with self.lock:
            self.db.execute("begin immediate")
            try:
                out = fn(self.db)
            except BaseException:
                self.db.execute("rollback")
                raise
            self.db.execute("commit")
            return out

    def create(self, meta: dict) -> str:
        sid = uuid.uuid4().hex[:12]
        self.tx(lambda db: db.execute("insert into sessions (id, created_at, meta) values (?, ?, ?)",
                                      (sid, time.time(), json.dumps(meta))))
        return sid

    def row(self, sid: str) -> dict:
        r = self.db.execute("select meta, meter, owner, lease_until, restarts from sessions where id = ?",
                            (sid,)).fetchone()
        if r is None:
            raise KeyError(f"no session {sid}")
        return {"meta": json.loads(r[0]), "meter": json.loads(r[1]), "owner": r[2],
                "lease_until": r[3], "restarts": r[4]}

    def entries(self, sid: str) -> list[tuple[str, dict]]:
        rows = self.db.execute("select kind, data from entries where session_id = ? order by seq", (sid,))
        return [(kind, json.loads(data)) for kind, data in rows]

    def list(self) -> list[dict]:
        out = []
        for sid, created, meta, owner, until in self.db.execute(
                "select id, created_at, meta, owner, lease_until from sessions order by created_at"):
            entries = self.entries(sid)
            first = next((d["content"] for k, d in entries if k == "user"), "")
            state = fold(entries)
            out.append({"id": sid, "created_at": created, "status": state.status,
                        "first_prompt": first if isinstance(first, str) else "(compacted)",
                        "live": until > time.time() and not _dead_here(owner), "meta": json.loads(meta)})
        return out


@dataclass
class State:
    """What fold() rebuilds from the log."""
    messages: list = field(default_factory=list)
    todos: list = field(default_factory=list)
    open_calls: list = field(default_factory=list)   # tool calls of an unanswered round, in order
    started: dict = field(default_factory=dict)      # tool_use_id -> intent, for calls that began
    results: dict = field(default_factory=dict)      # tool_use_id -> tool_result block
    answers: dict = field(default_factory=dict)      # request_id -> final text
    requests: set = field(default_factory=set)       # every request_id ever submitted
    last_final: str = ""
    status: str = "new"                              # new | idle | needs_model | open_round


def fold(entries: list[tuple[str, dict]]) -> State:
    """Rebuild the conversation from the log. Messages are only ever *derived*; the same
    function serves `--resume`, `--list` and the tests."""
    s = State()
    round_results: dict = {}
    calls: list = []
    request = None
    for kind, data in entries:
        if kind == "user":
            s.messages.append({"role": "user", "content": data["content"]})
            request = data.get("request_id")
            if request:
                s.requests.add(request)
            s.status = "needs_model"
        elif kind == "assistant":
            s.messages.append({"role": "assistant", "content": data["content"]})
            calls = [b for b in data["content"] if b.get("type") == "tool_use"]
            round_results = {}
            s.started = {}
            s.status = "open_round" if calls else "needs_model"
        elif kind == "intent":
            s.started[data["id"]] = data
        elif kind == "result":
            round_results[data["tool_use_id"]] = data
            if calls and all(c["id"] in round_results for c in calls):
                # The round is complete: the results become one user turn, in call order.
                s.messages.append({"role": "user", "content": [round_results[c["id"]] for c in calls]})
                calls, round_results, s.started = [], {}, {}
                s.status = "needs_model"
        elif kind == "todos":
            s.todos = data["todos"]
        elif kind == "compaction":
            s.messages = data["messages"]
        elif kind == "final":
            s.last_final = data["text"]
            if request:
                s.answers[request] = data["text"]
            s.status = "idle"
    s.open_calls = calls
    s.results = round_results
    return s


class Session:
    """One session, held by one process at a time."""

    def __init__(self, store: Store, sid: str, owner: str | None = None):
        self.store, self.id, self.owner = store, sid, owner or me()
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None

    @classmethod
    def create(cls, store: Store, meta: dict | None = None, owner: str | None = None) -> "Session":
        session = cls(store, store.create(meta or {}), owner)
        session.acquire()
        return session

    @classmethod
    def open(cls, store: Store, sid: str, owner: str | None = None) -> "Session":
        session = cls(store, sid, owner)
        session.acquire()
        return session

    # --- the lease -------------------------------------------------------------------

    def acquire(self, now: float | None = None) -> None:
        def take(db):
            owner, until = db.execute("select owner, lease_until from sessions where id = ?", (self.id,)).fetchone()
            t = time.time() if now is None else now
            if owner not in (None, self.owner) and until > t and not _dead_here(owner):
                raise LeaseHeld(f"session {self.id} is held by {owner} for another {until - t:.0f}s")
            db.execute("update sessions set owner = ?, lease_until = ? where id = ?",
                       (self.owner, t + LEASE_SECONDS, self.id))
        self.store.tx(take)

    def heartbeat(self) -> None:
        """Keep the lease while long work (a slow command, a subagent) commits nothing."""
        def beat():
            while not self._stop.wait(LEASE_SECONDS / 3):
                self.store.tx(lambda db: db.execute(
                    "update sessions set lease_until = ? where id = ? and owner = ?",
                    (time.time() + LEASE_SECONDS, self.id, self.owner)))
        self._beat = threading.Thread(target=beat, daemon=True)
        self._beat.start()

    def close(self) -> None:
        self._stop.set()
        self.store.tx(lambda db: db.execute(
            "update sessions set owner = null, lease_until = 0 where id = ? and owner = ?", (self.id, self.owner)))

    # --- reading and writing ---------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self.store.row(self.id)["meta"]

    def set_meta(self, **changes) -> None:
        meta = {**self.meta, **changes}
        self.store.tx(lambda db: db.execute("update sessions set meta = ? where id = ?", (json.dumps(meta), self.id)))

    def state(self) -> State:
        return fold(self.store.entries(self.id))

    def meter(self) -> dict:
        return self.store.row(self.id)["meter"]

    def append(self, *entries: tuple[str, dict], meter: dict | None = None, finished: bool = False) -> None:
        """Write entries atomically, checking we still hold the lease in the same transaction.
        `finished` resets the crash counter: the session reached an answer."""
        def write(db):
            owner, = db.execute("select owner from sessions where id = ?", (self.id,)).fetchone()
            if owner != self.owner:
                raise LeaseHeld(f"lost the lease on session {self.id} to {owner}; stopping before writing")
            seq, = db.execute("select coalesce(max(seq), 0) from entries where session_id = ?", (self.id,)).fetchone()
            for i, (kind, data) in enumerate(entries, 1):
                db.execute("insert into entries values (?, ?, ?, ?, ?)",
                           (self.id, seq + i, kind, json.dumps(data), time.time()))
            updates = ["lease_until = ?"]
            values: list = [time.time() + LEASE_SECONDS]
            if meter is not None:
                updates.append("meter = ?")
                values.append(json.dumps(meter))
            if finished:
                updates.append("restarts = 0")
            db.execute(f"update sessions set {', '.join(updates)} where id = ?", (*values, self.id))
        self.store.tx(write)

    def count_restart(self, force: bool = False) -> int:
        """Called when a resume finds unfinished work. Raises Parked past the limit."""
        def bump(db):
            n, = db.execute("select restarts from sessions where id = ?", (self.id,)).fetchone()
            n = 0 if force else n
            if n >= RESTART_LIMIT:
                raise Parked(f"session {self.id} was resumed {n} times without finishing. Read what it was "
                             f"doing (--show), then resume with --force if it's safe to try again.")
            db.execute("update sessions set restarts = ? where id = ?", (n + 1, self.id))
            return n + 1
        return self.store.tx(bump)
