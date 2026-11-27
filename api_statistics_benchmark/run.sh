#!/usr/bin/env bash
# Legacy Buffer Python API statistics: real-NPU overhead of the statistics hooks.
#
# Each pair runs the same package twice: statistics off (TRMT_LOG_ENABLE=false,
# where the decorator is an identity function) and statistics on. No second
# repository is required.
set -euo pipefail

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$DIR/.." && pwd)
PACKAGE=${DEEPEP_PACKAGE:-$REPO_ROOT/deep_ep}
WORK=$(mktemp -d "${TMPDIR:-/tmp}/deepep-api-statistics-benchmark.XXXXXX")
RESULT=$WORK/results
CARDS=${CARDS:-0,1,2,3}
ROUNDS=${ROUNDS:-10}
NPROC=${NPROC:-4}
TRMT_LOG_PATH=${TRMT_LOG_PATH:-$DIR/artifacts/log}
EXPECT_COMMIT=${EXPECT_COMMIT:-}
trap 'rm -rf "$WORK"' EXIT

[ -d "$PACKAGE" ] || { echo "package not found: $PACKAGE (set DEEPEP_PACKAGE)" >&2; exit 1; }
if ! compgen -G "$PACKAGE/_C*.so" >/dev/null || ! compgen -G "$PACKAGE/_clos_C*.so" >/dev/null; then
  cat >&2 <<EOF
missing native build: $PACKAGE/_C*.so and/or $PACKAGE/_clos_C*.so
build once at the repository root:
  python3 -m pip install -v --force-reinstall --no-build-isolation --no-deps .
then rerun: bash $DIR/run.sh
EOF
  exit 1
fi

export TRMT_LOG_PATH
export TRMT_LOG_INTERVAL=${TRMT_LOG_INTERVAL:-10}
export TRMT_LOG_CONSOLE_OUTPUT=false
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export HCCL_NPU_SOCKET_PORT_RANGE=auto
HOST_NAME=$(hostname)
getent hosts "$HOST_NAME" >/dev/null 2>&1 || printf '127.0.0.1 %s\n' "$HOST_NAME" >> /etc/hosts

COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)
if [ -n "$EXPECT_COMMIT" ] && [ "$COMMIT" != "$EXPECT_COMMIT" ]; then
  echo "unexpected commit: $COMMIT, expected $EXPECT_COMMIT" >&2
  exit 1
fi

mkdir -p "$WORK/pkg/deep_ep" "$RESULT" "$TRMT_LOG_PATH"
cp "$DIR/benchmark.py" "$WORK/benchmark.py"
cp -aL "$PACKAGE/." "$WORK/pkg/deep_ep/"
find "$WORK/pkg" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

for round in $(seq 1 "$ROUNDS"); do
  if [ $((round % 2)) -eq 1 ]; then
    label=off
    enable=false
  else
    label=on
    enable=true
  fi

  output=$RESULT/$label-$round.json
  echo "=== round $round: statistics $label ==="
  (
    cd "$WORK"
    PYTHONPATH="$WORK/pkg:$WORK" \
    TRMT_LOG_ENABLE="$enable" \
    ASCEND_RT_VISIBLE_DEVICES="$CARDS" \
    DEEPEP_SHMEM_ENDPOINT="tcp://127.0.0.1:$((19300 + round))" \
    GLOO_SOCKET_IFNAME=lo \
    PYTHONUNBUFFERED=1 \
    python3 -m torch.distributed.run \
      --standalone --master-addr=127.0.0.1 --master-port=$((19500 + round)) \
      --nnodes=1 --nproc-per-node="$NPROC" --max-restarts=0 \
      benchmark.py
  ) | tee "$output"
done

python3 - "$RESULT" "$DIR/api_overhead.md" "$COMMIT" "$CARDS" "$NPROC" <<'PY'
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])
report = Path(sys.argv[2])
commit, cards, nproc = sys.argv[3], sys.argv[4], sys.argv[5]
rows = []
for path in sorted(root.glob("*.json")):
    label = path.name.split("-")[0]
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    offset = 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            break
        value, offset = decoder.raw_decode(text, start)
        rows.append({"variant": label, **value})

apis = ("get_dispatch_layout", "dispatch_fresh", "dispatch_cached",
        "intranode_combine")


def values(variant, rank, api, metric):
    return [row[f"{api}_{metric}"] for row in rows
            if row["variant"] == variant and row["rank"] == rank]


table = []
for variant in ("off", "on"):
    for rank in sorted({row["rank"] for row in rows}):
        for api in apis:
            means = values(variant, rank, api, "mean")
            p90s = values(variant, rank, api, "p90")
            if means and p90s:
                table.append((variant, str(rank), api, str(len(means)),
                              f"{statistics.fmean(means):.1f}",
                              f"{statistics.fmean(p90s):.1f}"))

for rank in sorted({row["rank"] for row in rows}):
    for api in apis:
        off_mean = values("off", rank, api, "mean")
        on_mean = values("on", rank, api, "mean")
        off_p90 = values("off", rank, api, "p90")
        on_p90 = values("on", rank, api, "p90")
        if not all((off_mean, on_mean, off_p90, on_p90)):
            continue
        off_mean_value = statistics.fmean(off_mean)
        on_mean_value = statistics.fmean(on_mean)
        off_p90_value = statistics.fmean(off_p90)
        on_p90_value = statistics.fmean(on_p90)
        count = str(min(len(off_mean), len(on_mean)))
        table.append(("delta", str(rank), api, count,
                      f"{on_mean_value - off_mean_value:+.1f}",
                      f"{on_p90_value - off_p90_value:+.1f}"))
        table.append(("%", str(rank), api, count,
                      f"{(on_mean_value - off_mean_value) / off_mean_value * 100:+.2f}%",
                      f"{(on_p90_value - off_p90_value) / off_p90_value * 100:+.2f}%"))

header = ("Variant", "Rank", "API", "Rounds", "Mean (ns)", "P90 (ns)")
widths = [max(len(header[index]), *(len(row[index]) for row in table))
          for index in range(len(header))]
lines = [
    "# Legacy Buffer Python API statistics overhead", "",
    f"Commit: `{commit}` · variant: statistics **off** (`TRMT_LOG_ENABLE=false`) "
    f"vs **on** · cards: `{cards}` · EP: `{nproc}`",
    "Shape: token/rank `4096` · hidden `2816` · BF16 · top-k `8` · experts `256`",
    "Protocol: alternating off/on rounds; 10 warmup + 1000 measured calls per round; "
    "arithmetic mean of per-round means.",
    "Real NPU only. Event waits happen after `end_ns` and are outside the measured interval.",
    "",
    "| " + " | ".join(header[index].ljust(widths[index])
                       for index in range(len(header))) + " |",
    "| " + " | ".join(":" + "-" * (widths[index] - 1)
                       for index in range(len(header))) + " |",
]
for row in table:
    lines.append("| " + " | ".join(
        row[index].ljust(widths[index]) if index < 3 else row[index].rjust(widths[index])
        for index in range(len(header))) + " |")
report.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

cat "$DIR/api_overhead.md"
