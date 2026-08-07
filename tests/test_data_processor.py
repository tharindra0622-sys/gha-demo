import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processor import process_pipeline


def test_process_pipeline():
    """Runs the full pipeline. Expected to fail deep inside
    compute_average_score() due to a string sneaking through
    clean_records() undetected."""
    result = process_pipeline()
    assert isinstance(result, float)
    assert result > 0
