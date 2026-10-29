"""Tests for the eval harness. They use the scripted fakes, so no API key is needed."""
import shutil
from types import SimpleNamespace

import pytest

import evals
import fakes


def config(**overrides):
    base = dict(agent=str(evals.DEFAULT_AGENT), model="claude-opus-5-5", effort=None, max_turns=20,
                no_subagents=False, system_extra="", no_sandbox=True, parallel=4, trials=1)
    return SimpleNamespace(**{**base, **overrides})


@pytest.fixture(scope="module")
def cases():
    return evals.load_cases()


def test_cases_load(cases):
    assert len(cases) == 8
    assert {c.difficulty for c in cases} == {"easy", "medium", "hard"}
    assert all(c.task and (c.path / "hidden").is_dir() and (c.path / "solution").is_dir() for c in cases)


def test_every_case_is_sound(cases, capsys):
    assert evals.validate(cases), capsys.readouterr().out


def test_agent_cannot_plant_its_own_hidden_tests(cases):
    case = next(c for c in cases if c.name == "weekend")
    workdir = evals.fresh_repo(case)
    (workdir / ".eval_hidden").mkdir()
    (workdir / ".eval_hidden/test_fake.py").write_text("def test_ok():\n    assert True\n")
    passed, _ = evals.run_hidden_tests(workdir, case)
    assert not passed            # the planted test was replaced by the real hidden one
    shutil.rmtree(workdir)


def test_oracle_passes_everything(cases):
    results = evals.run(cases, config(), fakes.factory("oracle", cases))
    assert evals.summarize(results)["pass_rate"] == 1.0


def test_giving_up_fails_everything(cases):
    results = evals.run(cases, config(), fakes.factory("noop", cases))
    s = evals.summarize(results)
    assert s["pass_rate"] == 0.0 and s["errors"] == 0


def test_claiming_a_fix_is_not_a_fix(cases):
    # The flaky fake always *says* it fixed the bug. Only the hidden tests decide.
    hard = [c for c in cases if c.name == "retry"]
    results = evals.run(hard, config(trials=6), fakes.factory("flaky", hard, seed=3))
    assert all(r["final"] == "Fixed the bug." for r in results)
    assert 0 < sum(r["passed"] for r in results) < 6


def test_same_seed_same_results(cases):
    a = evals.run(cases, config(), fakes.factory("flaky", cases, seed=7))
    b = evals.run(cases, config(), fakes.factory("flaky", cases, seed=7))
    assert [r["passed"] for r in a] == [r["passed"] for r in b]


def test_a_crashing_model_is_a_failed_attempt_not_a_crashed_eval(cases):
    class Broken:
        messages = property(lambda self: self)

        def create(self, **_):
            raise RuntimeError("API down")

    results = evals.run(cases[:2], config(), lambda case, trial: Broken())
    assert all(r["error"] == "RuntimeError: API down" and not r["passed"] for r in results)


def test_variants_change_the_request(cases):
    seen = []

    class Recorder(fakes.GivesUp):
        def create(self, **request):
            seen.append(request)
            return super().create(**request)

    case = cases[:1]
    evals.run(case, config(no_subagents=True, system_extra="Always write a test {first}."), lambda c, t: Recorder())
    names = [t["name"] for t in seen[0]["tools"]]
    assert "task" not in names and "read_file" in names
    assert seen[0]["system"][0]["text"].endswith("Always write a test {first}.")


def test_summary_counts_flakiness():
    results = [{"case": "a", "passed": True, "error": None}, {"case": "a", "passed": False, "error": None},
               {"case": "b", "passed": True, "error": None}, {"case": "b", "passed": True, "error": None}]
    s = evals.summarize(results)
    assert (s["pass_rate"], s["solved_any"], s["solved_all"]) == (0.75, 2, 1)


def test_compare_calls_small_differences_noise():
    def results(passes, n):
        return [{"case": f"c{i}", "difficulty": "easy", "passed": i < passes, "error": None} for i in range(n)]
    assert "within noise" in evals.compare(results(6, 8), results(5, 8), "a", "b")
    assert "real difference" in evals.compare(results(40, 40), results(0, 40), "a", "b")
    assert "identical" in evals.compare(results(4, 8), results(4, 8), "a", "b")
