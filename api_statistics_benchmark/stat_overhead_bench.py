"""CPU micro-benchmark for the legacy Buffer API statistics decorator.

Runs without an NPU: the decorated functions only sleep, so the measured cost is
the decorator itself (off = identity decorator, on = worker started).
"""

import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_ep.utils.api_statistics import ApiStatisticsWorker  # noqa: E402


def plain_api(x):
    return x


def shaped_api(x):
    total = 0
    for value in range(60):
        total += value
    return total + x


def measure(function, iterations):
    clock = time.perf_counter_ns
    started = clock()
    for _ in range(iterations):
        function(1)
    return clock() - started


def main():
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    os.environ["TRMT_LOG_INTERVAL"] = "3600"
    os.environ["TRMT_LOG_ENABLE"] = "false"
    os.environ["TRMT_LOG_PATH"] = tempfile.mkdtemp() + "/log"
    disabled = ApiStatisticsWorker()
    os.environ["TRMT_LOG_ENABLE"] = "true"
    enabled = ApiStatisticsWorker()
    assert enabled.start(), "worker failed to start"

    cases = {}
    for label, api in (("empty", plain_api), ("shaped", shaped_api)):
        cases[label] = {
            "baseline": api,
            "off": disabled.decorate("intranode_dispatch")(api),
            "on": enabled.decorate("intranode_dispatch")(api),
        }

    samples = {}
    for _ in range(rounds):
        for label in cases:
            for mode in ("baseline", "off", "on"):
                samples.setdefault((label, mode), []).append(
                    measure(cases[label][mode], iterations) / iterations)

    enabled.stop()

    header = f"{'case':8} {'mode':9} {'median ns':>10} {'min ns':>10} {'delta ns':>10}"
    print(header)
    print("-" * len(header))
    for label in cases:
        baseline = statistics.median(samples[(label, "baseline")])
        for mode in ("baseline", "off", "on"):
            values = samples[(label, mode)]
            median_value = statistics.median(values)
            print(f"{label:8} {mode:9} {median_value:10.1f} {min(values):10.1f} "
                  f"{median_value - baseline:+10.1f}")


if __name__ == "__main__":
    main()
