"""Execute one candidate inside a disposable container."""

import base64
import json
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    candidate = Path(directory) / "candidate.py"
    candidate.write_bytes(base64.b64decode(sys.argv[1]))
    try:
        functions = runpy.run_path(str(candidate))
        function = functions[sys.argv[2]]
        for index, (args, expected) in enumerate(json.loads(sys.argv[3]), 1):
            actual = json.dumps(function(*args), sort_keys=True)
            if actual != json.dumps(expected, sort_keys=True):
                raise AssertionError(f"Failed case {index}")
    except SystemExit as error:
        raise RuntimeError("Candidate exited before completing the tests") from error
