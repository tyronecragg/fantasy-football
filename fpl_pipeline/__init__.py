"""Python replacement for the Fantasy Premier League workbook.

Stages: ingest -> markets -> team_model -> players (master DataFrame) -> optimisation.
Every stage snapshots its DataFrame to outputs/ for inspection. See PIPELINE_MAP.md.
"""
