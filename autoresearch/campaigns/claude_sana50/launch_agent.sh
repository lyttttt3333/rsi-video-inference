#!/usr/bin/env bash
set -euo pipefail
unset CLAUDE_CONFIG_DIR
exec claude -p \
  --model opus \
  --effort high \
  --autocompact 300k \
  --dangerously-skip-permissions \
  --name claude-sana-50 \
  --verbose \
  --output-format stream-json \
  "Read CLAUDE.md, POLICY.md, and AGENT_PROMPT.md completely. Begin the independent 50-round SANA campaign immediately, start Round 1 now, and continue autonomously through all 50 rounds without waiting for user confirmation." \
  > claude-session.log 2>&1

