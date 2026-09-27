"""Reserve shared GPU-hours and configure or launch draft Harbor tasks.

Reservations are conservative: a launched allocation is charged in full, even
if it fails. There is no automatic refund, so retries cannot silently overrun
the campaign. Only launches through this controller are accounted for.
"""
import argparse
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / 'benchmark/tasks'
STATE = Path(__file__).with_name('campaign-state.json')
TASKS = {'h3': 'minimax-h3-hot-inference', 'sana': 'sana-video2-hot-inference'}
LIMIT = 12 * 3600


def seconds(hours):
    value = Decimal(hours) * 3600
    if value != int(value) or value <= 0:
        raise ValueError('Allocation must be positive, with whole-second precision')
    return int(value)


def validate_allocation(allocation):
    if set(allocation) != set(TASKS) or any(type(x) is not int or x <= 0 for x in allocation.values()):
        raise ValueError('Both allocations must be positive integer seconds')
    if sum(allocation.values()) > LIMIT:
        raise ValueError('Combined allocations exceed 12 GPU-hours')


def configure_task(path, timeout):
    text = path.read_text()
    text = re.sub(r'(?m)^timeout_sec = \d+ # campaign-agent-budget\n', '', text)
    text = text.replace('[agent]\n', f'[agent]\ntimeout_sec = {timeout} # campaign-agent-budget\n')
    text = re.sub(r'TASK_BUDGET_SECS = "[^"]*"', f'TASK_BUDGET_SECS = "{timeout}"', text)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    configure = sub.add_parser('configure')
    configure.add_argument('--h3-hours', required=True)
    configure.add_argument('--sana-hours', required=True)
    sub.add_parser('status')
    launch = sub.add_parser('launch')
    launch.add_argument('task', choices=TASKS)
    launch.add_argument('harbor_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Serializes concurrent launch attempts on the shared host filesystem.
    with STATE.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(STATE.read_text()) if STATE.exists() else {'limit_gpu_seconds': LIMIT, 'allocation': None, 'launched': {}}
        if args.command == 'status':
            print(json.dumps(state, indent=2))
            return
        if args.command == 'configure':
            if state['launched']:
                raise SystemExit('Cannot reallocate this campaign after a launch; no automatic budget reset')
            allocation = {'h3': seconds(args.h3_hours), 'sana': seconds(args.sana_hours)}
            validate_allocation(allocation)
            for key, timeout in allocation.items():
                configure_task(PACKAGES / TASKS[key] / 'task.toml', timeout)
            state['allocation'] = allocation
        else:
            allocation = state['allocation']
            if allocation is None:
                raise SystemExit('Budget has not been allocated')
            validate_allocation(allocation)
            if args.task in state['launched']:
                raise SystemExit('This task allocation has already been charged; no unaccounted retries')
            # Restrict CLI overrides to agent identity, not resource/time/trial
            # multipliers. Raw harbor runs are outside this bookkeeping system.
            extra = args.harbor_args
            if extra[:1] == ['--']:
                extra = extra[1:]
            allowed = {'-a', '--agent', '-m', '--model', '--ak', '--agent-kwarg'}
            if len(extra) % 2 or any(extra[i] not in allowed for i in range(0, len(extra), 2)):
                raise SystemExit('Only agent/model/agent-kwarg option-value pairs are permitted')
            path = PACKAGES / TASKS[args.task]
            import tomllib
            config = tomllib.loads((path / 'task.toml').read_text())
            if config['metadata'].get('status') != 'ready':
                raise SystemExit('Draft task cannot launch: complete validation, scoring and assets, then mark ready')
            if config['environment']['gpus'] != 1 or config['agent']['timeout_sec'] != allocation[args.task]:
                raise SystemExit('Task resources differ from campaign allocation')
            command = ['harbor', 'run', '-p', str(path), '-e', 'modal', '-y', *extra]
            state['launched'][args.task] = {'reserved_gpu_seconds': allocation[args.task], 'command': command}
        STATE.write_text(json.dumps(state, indent=2) + '\n')
    if args.command == 'launch':
        raise SystemExit(subprocess.call(command))


if __name__ == '__main__':
    main()
