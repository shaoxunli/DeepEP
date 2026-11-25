#!/usr/bin/env python3
"""Extract allocator calls inside the probe's CPU markers, without double-counting nesting.

Pass exported trace_view.json files, not profiler summary JSON files.
API spellings are kept separate: do not sum wrapper and underlying API counts.
"""

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path


def kind(name):
    normalized = name.lower().replace("_", "").replace("::", "")
    if "recordevent" in normalized or normalized == "eventrecord":
        return "record"
    if "queryeventstatus" in normalized:
        return "query"
    if name == "empty_tensor":
        return "allocation"
    return None


def extract(events, source):
    scopes = [
        e
        for e in events
        if e.get("ph") == "X"
        and e.get("name", "").startswith("allocator_probe/")
        and e["name"].endswith(("/release_previous", "/fc1"))
    ]
    calls = [e for e in events if e.get("ph") == "X" and kind(e.get("name", ""))]
    rows = []
    for scope in scopes:
        begin, end = scope["ts"], scope["ts"] + scope["dur"]
        totals = {}
        for call in calls:
            # Allocator work here is synchronous Host work on the issuing thread.
            if (call.get("pid"), call.get("tid")) != (scope.get("pid"), scope.get("tid")):
                continue
            if begin <= call["ts"] and call["ts"] + call["dur"] <= end:
                total = totals.setdefault(call["name"], [0, Decimal(0)])
                total[0] += 1
                total[1] += call["dur"]
        if not totals:
            totals["NO_MATCHING_EVENTS"] = [0, Decimal(0)]
        for name, (count, duration) in totals.items():
            rows.append(
                dict(
                    source=source,
                    scope=scope["name"],
                    scope_us=float(scope["dur"]),
                    kind=kind(name) or "none",
                    api=name,
                    count=count,
                    api_sum_us=float(duration),
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.traces:
        with path.open(encoding="utf-8") as f:
            data = json.load(f, parse_float=Decimal)
        events = data if isinstance(data, list) else data["traceEvents"]
        extracted = extract(events, str(path))
        if not extracted:
            raise RuntimeError(f"No probe CPU scopes found in {path}")
        rows.extend(extracted)
        del data, events
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    print("NO_MATCHING_EVENTS does not prove zero overhead: verify profiler API visibility.")


if __name__ == "__main__":
    main()
