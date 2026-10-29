"""Scripted stand-ins for the model, so the eval pipeline can be run and tested without an
API key. They act through the real agent loop and tools; only the model is fake.

    oracle  reads the buggy file, writes the reference fix, says it's done (always passes)
    flaky   like oracle, but sometimes "fixes" nothing and still says it's done, more often
            on harder cases. Seeded by case and trial, so a run is reproducible.
    noop    gives up straight away (always fails)
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

from anthropic.types import TextBlock, ToolUseBlock, Usage

SUCCESS_RATE = {"easy": 0.9, "medium": 0.6, "hard": 0.3}


def _response(stop_reason, content, messages):
    size = len(json.dumps(messages, default=str)) // 4
    usage = Usage(input_tokens=size, output_tokens=80, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop_reason, content=content, usage=usage)


class ScriptedFixer:
    def __init__(self, case, works: bool):
        self.case, self.works = case, works
        self.messages = self       # so the agent can call client.messages.create(...)

    def create(self, **request):
        turn = sum(m["role"] == "assistant" for m in request["messages"])
        source = "solution" if self.works else "repo"
        files = sorted(p for p in (self.case.path / "solution").rglob("*") if p.is_file())
        rel = [p.relative_to(self.case.path / "solution") for p in files]
        if turn == 0:
            calls = [ToolUseBlock(type="tool_use", id=f"r{i}", name="read_file", input={"path": str(r)})
                     for i, r in enumerate(rel)]
            return _response("tool_use", [TextBlock(type="text", text="Let me look."), *calls], request["messages"])
        if turn == 1:
            calls = [ToolUseBlock(type="tool_use", id=f"w{i}", name="write_file",
                                  input={"path": str(r), "content": (self.case.path / source / r).read_text()})
                     for i, r in enumerate(rel)]
            return _response("tool_use", calls, request["messages"])
        return _response("end_turn", [TextBlock(type="text", text="Fixed the bug.")], request["messages"])


class GivesUp:
    def __init__(self):
        self.messages = self

    def create(self, **request):
        return _response("end_turn", [TextBlock(type="text", text="I couldn't find the bug.")], request["messages"])


def factory(kind: str, cases, seed: int = 0):
    def make_client(case, trial):
        if kind == "oracle":
            return ScriptedFixer(case, works=True)
        if kind == "flaky":
            rng = random.Random(f"{seed}-{case.name}-{trial}")
            return ScriptedFixer(case, works=rng.random() < SUCCESS_RATE[case.difficulty])
        return GivesUp()
    return make_client
