"""Print the odds provenance manifest: what is real vs synthetic right now, per market.

    python tools/odds_status.py

Reads sportsbet/_provenance.json (fpl_pipeline/provenance.py). Markets are stamped by the tools
that write them: build_synthetic_gw (synthetic), betway.py (real), bet365/ladbrokes cards. A market
with no entry shows 'unknown' - run tools/betway.py to populate it."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import provenance  # noqa: E402

MARK = {"real": "[REAL]", "synthetic": "[SYNTH]", "derived": "[deriv]", "unknown": "[ ?? ]"}


def main():
    doc = provenance.status()
    markets = doc.get("markets", {})
    print(f"\nOdds provenance - GW{doc.get('gw')}  (updated {doc.get('updated')})\n")
    # show every known sportsbet market in a stable order, even if unstamped
    for fname, label in provenance.FRIENDLY.items():
        e = markets.get(fname)
        st = e["state"] if e else "unknown"
        src = f"{e['source']}" + (f" - {e['detail']}" if e.get("detail") else "") if e else "no entry (run betway)"
        print(f"  {MARK.get(st, st):<8} {label:<18} {src}")
    real = sum(1 for e in markets.values() if e.get("state") == "real")
    synth = [provenance.FRIENDLY.get(f, f) for f, e in markets.items() if e.get("state") == "synthetic"]
    print(f"\n  {real} market(s) real; still synthetic: {', '.join(synth) or 'none'}")
    print("  --gw archiving should record only real markets (provenance.is_real).\n")


if __name__ == "__main__":
    main()
