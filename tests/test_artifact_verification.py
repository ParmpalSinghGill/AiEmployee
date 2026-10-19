"""Ground-truth artifact checks: cross-checking an agent's file claims against disk.

Regression coverage for a real failure observed in a live run: an agent's final
text answer quoted a byte count/PNG size from an earlier, superseded tool call
alongside the final one, and the LLM verifier rejected the (actually correct)
step because the prose was internally inconsistent. audit_artifacts measures the
real file directly so the verifier gets facts instead of trusting the agent's
self-reported arithmetic.
"""

from __future__ import annotations

import struct
from pathlib import Path

from ai_employee.verification.artifacts import audit_artifacts, requested_days_from_task
from ai_employee.verification.verifier import Verifier


def _write_png(path: Path, width: int, height: int) -> None:
    # Minimal valid PNG header (signature + IHDR) — enough for our header-only reader.
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    path.write_bytes(sig + b"\x00\x00\x00\x0dIHDR" + ihdr_data)


def test_no_files_mentioned_does_not_apply() -> None:
    audit = audit_artifacts(output="The result looks good.", workspace=Path("."))
    assert audit["applies"] is False
    assert audit["issues"] == []


def test_correct_byte_and_line_claims_produce_no_issues(tmp_path: Path) -> None:
    csv = tmp_path / "aapl_last_30d.csv"
    csv.write_text("Date,Close\n2026-08-12,302.25\n2026-08-13,305.26\n")
    output = f"Saved `{csv.name}` — {csv.stat().st_size} bytes, 3 lines."
    audit = audit_artifacts(output=output, workspace=tmp_path)
    assert audit["applies"] is True
    assert audit["issues"] == []
    assert audit["checked"][0]["actual_bytes"] == csv.stat().st_size


def test_wrong_byte_count_claim_is_flagged(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n")
    wrong = csv.stat().st_size + 3
    audit = audit_artifacts(output=f"Wrote {csv.name}, {wrong} bytes.", workspace=tmp_path)
    assert audit["applies"] is True
    assert any(str(wrong) in issue for issue in audit["issues"])


def test_wrong_line_count_claim_is_flagged(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    audit = audit_artifacts(output=f"{csv.name} has 99 lines.", workspace=tmp_path)
    assert any("99" in issue for issue in audit["issues"])


def test_correct_png_dimensions_produce_no_issue(tmp_path: Path) -> None:
    png = tmp_path / "chart.png"
    _write_png(png, 1391, 753)
    audit = audit_artifacts(output=f"{png.name} is 1391x753.", workspace=tmp_path)
    assert audit["issues"] == []
    assert audit["checked"][0]["actual_dimensions"] == (1391, 753)


def test_stale_png_dimension_claim_is_flagged_not_silently_passed(tmp_path: Path) -> None:
    """This is exactly the observed bug: the agent's final prose quoted an earlier
    (693px-tall) chart render alongside the final (753px-tall) one. The file on disk
    is the final, correct 753px version — audit_artifacts must catch the mismatch
    against the *actual* file rather than trust either number blindly."""
    png = tmp_path / "chart.png"
    _write_png(png, 1391, 753)
    audit = audit_artifacts(
        output=f"`file` reported {png.name} as 1391 x 693, then we re-saved it.",
        workspace=tmp_path,
    )
    assert any("693" in issue and "753" in issue for issue in audit["issues"])


def test_missing_file_claimed_as_saved_is_flagged(tmp_path: Path) -> None:
    audit = audit_artifacts(output="Saved output.csv to the workspace.", workspace=tmp_path)
    assert audit["applies"] is True
    assert any("does not exist" in issue for issue in audit["issues"])


def test_mentioned_but_not_claimed_missing_file_is_not_flagged(tmp_path: Path) -> None:
    # A suggestion/plan to write a file later shouldn't be treated as a false claim.
    audit = audit_artifacts(output="Next I will write report.csv.", workspace=tmp_path)
    assert audit["issues"] == []


def test_requested_days_from_task_parses_relative_ranges() -> None:
    assert requested_days_from_task("download Gold 1 minute data for the last 1 year") == 365
    assert requested_days_from_task("fetch the past 30 days of trades") == 30
    assert requested_days_from_task("get the last 6 months of prices") == 180
    assert requested_days_from_task("write a hello world script") is None


def test_short_date_span_against_requested_range_is_flagged(tmp_path: Path) -> None:
    """Regression for a real failure mode: Yahoo Finance caps 1-minute bars at ~7
    days regardless of the requested period. A fetch step should not be able to
    silently pass off 7 days as satisfying a 'last 1 year' request."""
    csv = tmp_path / "gold_1min.csv"
    rows = ["Datetime,Close"]
    import datetime as dt

    start = dt.date(2026, 9, 1)
    for i in range(7):
        rows.append(f"{start + dt.timedelta(days=i)} 09:30:00,2000.{i}")
    csv.write_text("\n".join(rows) + "\n")

    audit = audit_artifacts(
        output=f"Saved {csv.name} with Gold 1-minute data.",
        workspace=tmp_path,
        task="Download the Gold 1 minute data for the last 1 year",
    )
    assert audit["applies"] is True
    assert any("only spans" in issue for issue in audit["issues"])
    assert audit["checked"][0]["actual_span_days"] == 6


def test_full_requested_range_produces_no_date_issue(tmp_path: Path) -> None:
    csv = tmp_path / "gold_daily.csv"
    rows = ["Date,Close"]
    import datetime as dt

    start = dt.date(2025, 9, 1)
    for i in range(0, 366, 3):
        rows.append(f"{start + dt.timedelta(days=i)},2000.0")
    csv.write_text("\n".join(rows) + "\n")

    audit = audit_artifacts(
        output=f"Saved {csv.name} with Gold daily data.",
        workspace=tmp_path,
        task="Download the Gold daily data for the last 1 year",
    )
    assert audit["issues"] == []
    assert audit["checked"][0]["actual_span_days"] >= 360


def test_no_requested_range_skips_date_check(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("Date,Close\n2026-01-01,1.0\n2026-01-02,2.0\n")
    audit = audit_artifacts(
        output=f"Saved {csv.name}.", workspace=tmp_path, task="Analyze this dataset"
    )
    assert audit["issues"] == []
    assert "actual_span_days" not in audit["checked"][0]


async def test_verifier_includes_ground_truth_facts_in_prompt(brain) -> None:
    """Wiring check: Verifier.verify must actually forward artifact_audit content
    into the prompt it sends the brain, not just accept the parameter."""
    captured: dict = {}
    original = brain.structured_output

    async def spy(messages, schema, **kwargs):
        captured["messages"] = messages
        return await original(messages, schema, **kwargs)

    brain.structured_output = spy
    verifier = Verifier(brain)
    audit = {
        "applies": True,
        "issues": ["Output claims 'x.png' is 100x100, but the file is actually 200x200."],
        "checked": [{"file": "x.png", "actual_bytes": 123, "actual_dimensions": (200, 200)}],
    }
    await verifier.verify(task="make a chart", output="saved x.png at 100x100", artifact_audit=audit)
    user_content = captured["messages"][1].content
    assert "200x200" in user_content
    assert "GROUND-TRUTH" in user_content
