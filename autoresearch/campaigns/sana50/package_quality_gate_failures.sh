#!/usr/bin/env bash
set -euo pipefail
python /harbor/export_teachers.py --reference /campaign/quality-gate-failures-mp4
tar -C /campaign -cf /campaign/sana50-quality-gate-failures-rounds-20-24-28-29.tar quality-gate-failures-mp4
