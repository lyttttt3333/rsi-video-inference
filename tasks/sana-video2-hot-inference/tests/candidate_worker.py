#!/usr/bin/env python3
"""Untrusted candidate worker for the parent-controlled measurement protocol.

The parent sends exactly one request at a time.  The worker never receives the
held-out case list or the verifier's scoring code.  Control records are
prefixed so candidate stdout cannot forge them; candidate stdout/stderr are
redirected to /dev/null while user code runs.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys


def control(value: dict) -> None:
    print("__RSI_CONTROL__" + json.dumps(value, separators=(",", ":")), flush=True)


def load_module(path: Path):
    # Candidate submissions commonly carry a sibling `model/` package.  The
    # old in-process evaluator inserted the submission root into sys.path;
    # preserve that contract in the isolated worker as well.
    submission_root = str(path.parent.resolve())
    if submission_root not in sys.path:
        sys.path.insert(0, submission_root)
    spec = importlib.util.spec_from_file_location("candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()

    # Keep the protocol channel clean even when candidate code prints logs.
    devnull = open("/dev/null", "w")
    try:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            import torch
            module = load_module(args.submission / "candidate.py")
            runner = module.build(args.weights)
            torch.cuda.synchronize()
    except Exception as error:
        control({"event": "error", "error": f"{type(error).__name__}: {error}"})
        return 1
    finally:
        devnull.close()

    control({"event": "ready"})
    pending = None
    for line in sys.stdin:
        try:
            command = json.loads(line)
            operation = command.get("op")
            if operation == "warmup":
                with contextlib.redirect_stdout(open("/dev/null", "w")), contextlib.redirect_stderr(open("/dev/null", "w")):
                    output = runner.generate(dict(command["request"]))
                    torch.cuda.synchronize()
                    # Shape and finiteness are checked by the parent after a timed run.
                    del output
                control({"event": "warmup_done"})
            elif operation == "generate":
                with contextlib.redirect_stdout(open("/dev/null", "w")), contextlib.redirect_stderr(open("/dev/null", "w")):
                    pending = runner.generate(dict(command["request"]))
                    torch.cuda.synchronize()
                # The parent stops its trusted timer on this message.  Tensor
                # export is performed only after the timer has stopped.
                control({"event": "generated"})
            elif operation == "save":
                if pending is None:
                    raise RuntimeError("save without generate")
                with contextlib.redirect_stdout(open("/dev/null", "w")), contextlib.redirect_stderr(open("/dev/null", "w")):
                    tensors = {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in pending.items()
                    }
                    torch.save(tensors, Path(command["path"]))
                    del tensors, pending
                    pending = None
                control({"event": "saved"})
            elif operation == "stop":
                control({"event": "stopped"})
                return 0
            else:
                raise ValueError(f"Unknown worker operation: {operation!r}")
        except Exception as error:
            control({"event": "error", "error": f"{type(error).__name__}: {error}"})
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
