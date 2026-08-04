#!/usr/bin/env python3
"""Software-in-the-loop CENSUS of everything the companion puts on the FC wire.

`hil_standby_check.py` proves the STANDBY contract against a real FC, but on the
bench that FC is disarmed with no TX bound — so it can only ever exercise one of
the states that matter. This drives the REAL Pipeline + REAL ArduPilotBackend
against the UDP FakeArduCopter, which CAN be armed, put in an arbitrary flight
mode, and have its RC channels moved:

  1. STANDBY + DISARMED              — preflight on the bench
  2. STANDBY + ARMED                 — the flight-3 case: pilot flying manually,
                                       companion must be completely silent
  3. STANDBY + ARMED + FC in GUIDED  — orphaned engage; the ONE case where the
                                       companion deliberately commands a mode
  4. engage -> disengage             — the restore edge

Every outbound send is wrapped, not just the ones the contract checker consumes,
so the output is a full census: if the companion transmits anything at all in
STANDBY, it shows up here.

The same scenarios are asserted in tests/test_standby_wire_contract.py; this
script is the human-readable view of the same harness.

    python scripts/sil_standby_audit.py
Exit non-zero on any unexpected contract violation.
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from tests.fakes.sil_wire_audit import CONTROL_AFFECTING, run_scenario

STABILIZE, GUIDED_NOGPS = 0, 20
STANDBY_US, TRACK_US = 1000, 1500

SCENARIOS = [
    ("1. STANDBY + DISARMED (bench preflight)",
     dict(armed=False, fc_mode=STABILIZE, ch7_us=STANDBY_US, seconds=2.5), False),
    ("2. STANDBY + ARMED, FC in STABILIZE (pilot flying — flight-3 case)",
     dict(armed=True, fc_mode=STABILIZE, ch7_us=STANDBY_US, seconds=2.5), False),
    ("3. STANDBY + ARMED, FC already in GUIDED_NOGPS (orphaned engage)",
     dict(armed=True, fc_mode=GUIDED_NOGPS, ch7_us=STANDBY_US, seconds=2.5), True),
    ("4. engage -> disengage cycle (restore edge)",
     dict(armed=True, fc_mode=STABILIZE, ch7_us=STANDBY_US, engage_us=TRACK_US,
          seconds=4.5), False),
]


def main() -> int:
    failed = False
    for label, kw, expect_orphan_recovery in SCENARIOS:
        audit = run_scenario(**kw)
        print()
        print("=" * 78)
        print(label)
        print("=" * 78)
        if not audit.census:
            print("    (nothing transmitted)")
        for kind in sorted(audit.census):
            flag = "  <-- TOUCHES FLIGHT CONTROL" if kind in CONTROL_AFFECTING else ""
            print(f"    {kind:<32} {audit.census[kind]:5d}{flag}")
        for kind, mode, armed in audit.control_events():
            print(f"      ! {kind} while switch={mode} armed={armed}")

        if audit.checker.passed:
            print("    contract: PASS")
            continue
        kinds = {v.kind for v in audit.checker.violations}
        if (expect_orphan_recovery and kinds == {"STANDBY-no-mode-cmd"}
                and audit.count("do_set_mode") == 1):
            # Deliberate, documented exception — see recover_orphaned_mode.
            print("    contract: EXPECTED BREACH (orphan-recovery handback only)")
            for v in audit.checker.violations:
                print(f"      {v.detail}")
            continue
        print("    contract: FAIL")
        print(audit.checker.report())
        failed = True

    print()
    print("Scenario 3's breach is BY DESIGN: recover_orphaned_mode hands the sticks")
    print("back after a restart orphaned an engage. It is the only path that commands")
    print("a flight mode while the engage switch reads STANDBY.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
