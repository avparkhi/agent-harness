"""Step 3 (unchanged in step 6 except that memory can live on a remote machine): manage what's in the context window.

Three ideas, one file:
    memory      a HARNESS.md file (like CLAUDE.md) read once at startup into the system prompt
    caching     mark the stable prefix so repeat requests are billed at a fraction of the price
    compaction  when the conversation gets long, replace it with a summary and start fresh

They all follow from one fact: the API is stateless, so every request resends everything,
and the model only knows what's in that one request.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

# --- memory -------------------------------------------------------------------------------

MEMORY_FILE = "HARNESS.md"
USER_MEMORY = Path("~/.harness/HARNESS.md")


def load_memory(root, read_project=None) -> str:
    """User-level memory first, then the project's. Read once per session, never re-read:
    the system prompt must stay byte-identical for caching (and for thinking blocks).
    read_project(name) reads the project file from somewhere else, such as a remote machine."""
    parts = []
    user = USER_MEMORY.expanduser()
    if user.is_file():
        parts.append(f"<memory source=\"user\" path=\"{user}\">\n{user.read_text().strip()}\n</memory>")
    path = f"{root}/{MEMORY_FILE}"
    if read_project:
        text = read_project(MEMORY_FILE)
    else:
        text = Path(path).read_text() if Path(path).is_file() else None
    if text:
        parts.append(f"<memory source=\"project\" path=\"{path}\">\n{text.strip()}\n</memory>")
    return "\n\n".join(parts)


def with_note(text: str | None, note: str) -> str:
    """The memory file's new text after adding one note."""
    text = text or "# Project memory\n"
    if "## Learned" not in text:
        text = text.rstrip() + "\n\n## Learned\n"
    return text.rstrip() + f"\n- {note.strip()}\n"


def remember(root: Path, note: str) -> str:
    """Append a note to the project's memory file. It takes effect next session."""
    path = Path(root) / MEMORY_FILE
    path.write_text(with_note(path.read_text() if path.exists() else None, note))
    return REMEMBERED


REMEMBERED = f"Saved to {MEMORY_FILE}. It will be in the system prompt from the next session."


# --- caching and cost -------------------------------------------------------------------

# Claude Opus 5.5, dollars per million tokens. A cache write costs 1.25x normal input;
# a cache read costs 0.05x. That 25x gap is why caching matters so much for agents.
PRICES = {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00}


@dataclass
class Meter:
    input: int = 0          # uncached input tokens
    cache_write: int = 0    # input tokens written to the cache this request
    cache_read: int = 0     # input tokens served from the cache
    output: int = 0
    requests: int = 0
    last_context: int = 0   # how big the conversation was at the last request
    history: list = field(default_factory=list)
    subagent_cost: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage) -> None:
        i = getattr(usage, "input_tokens", 0) or 0
        w = getattr(usage, "cache_creation_input_tokens", 0) or 0
        r = getattr(usage, "cache_read_input_tokens", 0) or 0
        o = getattr(usage, "output_tokens", 0) or 0
        self.input, self.cache_write, self.cache_read, self.output = (
            self.input + i, self.cache_write + w, self.cache_read + r, self.output + o)
        self.requests += 1
        # The next request will carry everything this one did, plus this response.
        self.last_context = i + w + r + o
        self.history.append((i, w, r, o))

    def absorb(self, child: Meter) -> None:
        """Add a subagent's spend to this meter. Its context size is its own, so not that."""
        with self.lock:
            self.input += child.input
            self.cache_write += child.cache_write
            self.cache_read += child.cache_read
            self.output += child.output
            self.requests += child.requests
            self.subagent_cost += child.cost

    @property
    def cost(self) -> float:
        return (self.input * PRICES["input"] + self.cache_write * PRICES["cache_write"]
                + self.cache_read * PRICES["cache_read"] + self.output * PRICES["output"]) / 1e6

    @property
    def cost_without_cache(self) -> float:
        return ((self.input + self.cache_write + self.cache_read) * PRICES["input"]
                + self.output * PRICES["output"]) / 1e6

    def summary(self) -> str:
        total_in = self.input + self.cache_write + self.cache_read
        hit = self.cache_read / total_in if total_in else 0
        return (f"{self.requests} requests | context now {self.last_context:,} tokens | "
                f"input {total_in:,} ({hit:.0%} from cache) | output {self.output:,} | "
                f"${self.cost:.4f} (${self.cost_without_cache:.4f} without caching)"
                + (f" | ${self.subagent_cost:.4f} of it in subagents" if self.subagent_cost else ""))


# --- compaction -------------------------------------------------------------------------

# From Anthropic's guidance for client-side compaction. The last sentence matters: the
# summary request still carries the tools, so without it the model sometimes calls one.
SUMMARY_PROMPT = """Summarize the transcript inside <summary></summary> tags. Include relevant information in the summary such that this conversation will be continued by a new context window without needing to redo work or be reprovided with relevant constraints or context. Be sure to preserve: (1) any difficulties or problems that came up, and how they were handled or resolved; (2) any possibilities, options, or approaches that were raised, tried, or set aside, and why; (3) anything that was asked for, decided, agreed, ruled out, or established as a preference, constraint, or boundary - stated exactly; (4) exactly where things stand now - what has been covered, settled, or completed so far; (5) anything still open, unresolved, promised, or expected to happen next; (6) specific details that would be hard to reconstruct - names, numbers, dates, exact wording, links or references - kept exactly. Be complete on these even at the cost of length; keep everything else concise. Weight the two voices differently: keep what the user said, asked for, shared, or established carefully and close to their own words; your own explanations and reasoning can be condensed much further, to what they concluded or produced - as long as nothing in the six items above is dropped. Do not call any tools while writing this summary; respond with text only."""

CONTINUATION = ("This session continues an earlier conversation that was summarized to free up "
                "context. Summary of everything so far:\n\n{summary}\n\n")


def with_summary_request(messages: list[dict]) -> list[dict]:
    """The messages for the summary call: the conversation exactly as is (so the cached
    prefix is reused and the summary is cheap), plus the summary instruction appended to
    the last user turn. The real history is not modified."""
    prompt = {"type": "text", "text": SUMMARY_PROMPT}
    last = messages[-1]
    if last["role"] == "assistant":          # between user turns: add a new user turn
        return [*messages, {"role": "user", "content": [prompt]}]
    content = last["content"]                # after tool results: join the same user turn
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content)
    return [*messages[:-1], {"role": "user", "content": [*blocks, prompt]}]


def extract_summary(content) -> str:
    text = "".join(b.text for b in content if b.type == "text")
    if "<summary>" in text and "</summary>" in text:
        text = text.split("<summary>", 1)[1].split("</summary>", 1)[0]
    return text.strip()
