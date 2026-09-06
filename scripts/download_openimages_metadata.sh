#!/usr/bin/env bash
set -Eeuo pipefail

repo=/home/ubuntu/robot/robot_ex1
cache="$repo/.cache/openimages-v7"
job="$repo/.cache/jobs/openimages_metadata_20260831"
mkdir -p "$cache" "$job"

printf '%s\n' "$$" > "$job/worker.pid"
printf '%s\n' "$(date -Is)" > "$job/started_at"
rm -f "$job/exit_code" "$job/completed_at"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$job/exit_code"
  printf '%s\n' "$(date -Is)" > "$job/completed_at"
  printf 'metadata job finished rc=%s at %s\n' "$rc" "$(date -Is)"
}
trap finish EXIT

old_pid=156732
if [[ -d "/proc/$old_pid" ]] && [[ "$(readlink -f "/proc/$old_pid/cwd" 2>/dev/null || true)" == "$repo" ]]; then
  printf 'waiting for legacy downloader pid=%s without interrupting it\n' "$old_pid"
  while kill -0 "$old_pid" 2>/dev/null; do
    size=$(stat -c %s "$cache/oidv7-train-annotations-human-imagelabels.csv" 2>/dev/null || printf 0)
    printf '%s legacy_pid=%s human_labels_bytes=%s\n' "$(date -Is)" "$old_pid" "$size"
    sleep 30
  done
  printf 'legacy downloader exited; validating and resuming only incomplete files\n'
fi

download_exact() {
  name=$1
  url=$2
  expected=$3
  path="$cache/$name"
  actual=$(stat -c %s "$path" 2>/dev/null || printf 0)
  if [[ "$actual" == "$expected" ]]; then
    printf 'complete %s bytes=%s\n' "$name" "$actual"
    return
  fi
  if (( actual > expected )); then
    backup="$path.oversize.$(date +%Y%m%dT%H%M%S)"
    mv -- "$path" "$backup"
    actual=0
    printf 'preserved oversize file as %s\n' "$backup"
  fi
  printf 'resume %s current=%s expected=%s\n' "$name" "$actual" "$expected"
  curl -LfsS \
    --retry 12 --retry-all-errors --retry-delay 5 \
    --connect-timeout 20 -C - -o "$path" "$url"
  actual=$(stat -c %s "$path")
  if [[ "$actual" != "$expected" ]]; then
    printf 'size mismatch %s actual=%s expected=%s\n' "$name" "$actual" "$expected" >&2
    return 1
  fi
  printf 'complete %s bytes=%s\n' "$name" "$actual"
}

download_exact \
  oidv7-class-descriptions-boxable.csv \
  https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv \
  12064
download_exact \
  oidv6-train-annotations-bbox.csv \
  https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv \
  2258447590
download_exact \
  oidv7-train-annotations-human-imagelabels.csv \
  https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv \
  2735816020

python3 - "$cache" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
checks = {
    "oidv7-class-descriptions-boxable.csv": 2,
    "oidv6-train-annotations-bbox.csv": 13,
    "oidv7-train-annotations-human-imagelabels.csv": 4,
}
for name, min_columns in checks.items():
    path = root / name
    with path.open("r", encoding="utf-8", newline="") as stream:
        first = next(csv.reader(stream))
    if len(first) < min_columns:
        raise RuntimeError(f"invalid CSV header/row in {path}: {first}")
    print(f"csv_ok {name} first_columns={len(first)}")
PY

printf '%s\n' "$(date -Is)" > "$job/metadata_ready"
