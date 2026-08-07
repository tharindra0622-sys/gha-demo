"""
data_processor.py

Simulates a small real-world data pipeline: load records, clean them,
then compute statistics. Contains a deliberate, subtle bug for testing
whether an AI diagnosis agent can trace a multi-level failure back to
its real root cause (not just read the last line of the traceback).
"""


def load_records():
    """Pretend this loaded rows from a CSV or API. One row is malformed
    on purpose — its 'score' field is a string instead of a number,
    simulating a real-world messy-data bug."""
    return [
        {"name": "build_1", "score": 82},
        {"name": "build_2", "score": 91},
        {"name": "build_3", "score": "95"},   # bug: should be int 95, not "95"
        {"name": "build_4", "score": 77},
    ]


def clean_records(records):
    """Supposed to normalize records before use. It filters out None
    scores, but does NOT catch the case where score is a string that
    looks like a number — so the bad value passes through unnoticed."""
    cleaned = []
    for r in records:
        if r["score"] is not None:
            cleaned.append(r)
    return cleaned


def compute_average_score(records):
    """Adds up all scores and divides by count. Works fine for numbers,
    but breaks if any score slipped through as a string."""
    total = 0
    for r in records:
        total += r["score"]   # <-- fails here: int + str
    return total / len(records)


def process_pipeline():
    """Entry point that ties the whole thing together."""
    records = load_records()
    cleaned = clean_records(records)
    return compute_average_score(cleaned)
