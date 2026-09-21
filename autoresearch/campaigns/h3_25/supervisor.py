"""Fail-closed ledger for the 25-round MiniMax-H3 public-three campaign."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
STATE_PATH = HERE / "state.json"
ROUND_LIMIT = 25
ROUND_RESERVE_SECONDS = 1680


def dump(value):
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(STATE_PATH)


parser = argparse.ArgumentParser()
commands = parser.add_subparsers(dest="command", required=True)
start = commands.add_parser("start")
start.add_argument("--round", type=int, required=True)
start.add_argument("--hypothesis", required=True)
finish = commands.add_parser("finish")
finish.add_argument("--round", type=int, required=True)
finish.add_argument("--job-id", required=True)
commands.add_parser("status")
args = parser.parse_args()

with (HERE / "state.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    state = json.loads(STATE_PATH.read_text())
    if args.command == "status":
        print(json.dumps(state, indent=2))
        raise SystemExit
    if args.round != state["next_round"]:
        raise SystemExit(f'Expected round {state["next_round"]}')

    if args.command == "start":
        if state["active"] is not None:
            raise SystemExit("A round is already active")
        if not 1 <= args.round <= ROUND_LIMIT:
            raise SystemExit("Round is outside 1..25")
        if state["reserved_gpu_seconds"] + ROUND_RESERVE_SECONDS > state["limit_gpu_seconds"]:
            raise SystemExit("25-round reservation cap would be exceeded")
        workspace = HERE / "workspace"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=workspace,
            check=True, capture_output=True, text=True).stdout
        if dirty:
            raise SystemExit("Commit the proposed source state before starting the round")
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=workspace,
            check=True, capture_output=True, text=True).stdout.strip()
        output = HERE / "runs" / f"round-{args.round:02d}"
        if output.exists():
            raise FileExistsError(output)
        state["reserved_gpu_seconds"] += ROUND_RESERVE_SECONDS
        state["active"] = {
            "round": args.round,
            "hypothesis": args.hypothesis,
            "source_commit": commit,
            "reserve_seconds": ROUND_RESERVE_SECONDS,
            "output": str(output),
            "started_at": time.time(),
        }
        dump(state)
        print(json.dumps(state["active"], indent=2))
        print(f"bash {HERE / 'run_round.sh'} {args.round}", flush=True)
        raise SystemExit

    active = state["active"]
    if active is None or active["round"] != args.round:
        raise SystemExit("Round is not active")
    query = subprocess.run(
        ["sacct", "-j", args.job_id,
         "--format=JobIDRaw,State,ElapsedRaw,ExitCode", "-n", "-P"],
        check=True, capture_output=True, text=True)
    rows = [line.split("|") for line in query.stdout.splitlines() if line.strip()]
    root = next((row for row in rows if row[0] == args.job_id), None)
    terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"}
    if root is None or root[1].split("+")[0] not in terminal:
        raise SystemExit("Job is missing or not terminal")
    report_path = Path(active["output"]) / "reward.json"
    result = (json.loads(report_path.read_text()) if report_path.exists()
              else {"invalid": 1, "reward": 0.0, "error": "missing report"})
    active.update(job_id=args.job_id, slurm_state=root[1],
                  elapsed_seconds=int(root[2]), result=result,
                  finished_at=time.time())
    if not result.get("invalid", 1) and result.get("reward", 0.0) > state["best"]["reward"]:
        state["best"] = {
            "round": args.round,
            "reward": result["reward"],
            "source_commit": active["source_commit"],
            "output": active["output"],
        }
    state["history"].append(active)
    state["active"] = None
    state["next_round"] += 1
    dump(state)
    print(json.dumps(active, indent=2))
