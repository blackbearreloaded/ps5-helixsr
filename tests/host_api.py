"""Create the public runtime and record all three output-head paths."""
import json
from pathlib import Path
import subprocess
import sys

runner, weights, report = sys.argv[1:]
result = subprocess.run([runner, weights], capture_output=True, text=True)
assert result.returncode == 0, result.stderr
payload = json.loads(result.stdout)
assert payload["result"] == "success" and payload["contexts"] == 3
assert payload["model_bytes"] == 1903872
assert payload["recorded"] and not payload["submitted"] and payload["in_flight_rejected"]
receipt = {"schema": 1, **payload, "application_views_bound": True,
           "model_repacked_and_uploaded_during_create": True,
           "execution_qualified": False,
           "scope": "public API context creation and command recording; no dispatch execution"}
Path(report).write_text(json.dumps(receipt, indent=2) + "\n")
print("public API creates and records all three output-head contexts")
