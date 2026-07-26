"""The golden test: parity mode must reproduce the frozen workbook cell-for-cell.

Meaningful only while inputs/ and the sportsbet CSVs are unchanged since the workbook
was last calculated — improved-mode gameweek runs mutate inputs/, after which this
compares against a stale reference and may legitimately fail.
"""
from fpl_pipeline import run, validate


def test_full_parity_against_workbook():
    master = run.run(parity_mode=True)
    report = validate.run(master)
    bad = report[report["status"] != "ok"]
    assert bad.empty, f"parity mismatches:\n{bad.to_string()}"
