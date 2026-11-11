"""List the warnings recorded for each processed session.

The summary table counts warnings; this prints them, grouped so that the ones
common to every session (dead channels, say) are separated from the ones
specific to a session, which are the interesting ones.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--only-unique", action="store_true",
                    help="hide warnings that every session shares")
    args = ap.parse_args()

    mans = {}
    for p in sorted(Path(args.out_root).glob("*/*_manifest.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        mans[m.get("session_id", p.stem)] = m
    if not mans:
        print("no manifests found")
        return 1

    counts = Counter(w for m in mans.values() for w in m.get("warnings", []))
    universal = {w for w, c in counts.items() if c == len(mans)}

    if universal and not args.only_unique:
        print(f"=== shared by all {len(mans)} sessions ===")
        for w in sorted(universal):
            print(f"  ! {w}")

    print("\n=== session-specific ===")
    for sid, m in mans.items():
        ws = [w for w in m.get("warnings", []) if w not in universal]
        qc = m.get("qc", {})
        head = (f"{sid}  [{m.get('status')}]  "
                f"led_sep={qc.get('led_separation', float('nan')):.1f}  "
                f"blanked={qc.get('n_blanked')}  "
                f"events={qc.get('n_laser_events')}")
        print(f"\n{head}")
        for w in ws:
            print(f"  ! {w}")
        if not ws:
            print("  (none)")

    r2 = {sid: m.get("qc", {}).get("regression_r2") for sid, m in mans.items()}
    print("\n=== isosbestic R^2 (median across ROIs) ===")
    for sid, v in r2.items():
        if v:
            vals = [x for x in v if x is not None]
            if vals:
                vals = sorted(vals)
                med = vals[len(vals) // 2]
                print(f"  {sid:30s} {med:.3f}  (max {max(vals):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
