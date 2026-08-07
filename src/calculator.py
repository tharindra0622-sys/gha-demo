import requests
#
# test run
# trigger new test run.
def fetch_status(url):
    """Deliberately uses a package NOT in requirements.txt,
    to test whether the LLM correctly diagnoses a missing dependency."""
    return requests.get(url).status_code
