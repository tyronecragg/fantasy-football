"""Collect bet365 odds by driving a real browser, then parse with tools/bet365.py.

    pip install playwright && playwright install chromium
    python tools/bet365_browser.py                       # Premier League, all fixtures
    python tools/bet365_browser.py --limit 3 --headed    # watch it work

Writes outputs/bet365_odds.csv, same shape as the HAR path.

WHY A BROWSER. Their content APIs are gated on X-Net-Sync-Term, a ~1660-char token their
obfuscated bundle mints client-side. Everything else was eliminated on 2026-08-16: TLS
fingerprinting (curl_cffi impersonation passes), geography (a HAR captured locally
succeeds), cookies (all 11 replayed), header set and order (copied verbatim, plus a fresh
X-Request-Id), compression (zero raw bytes), HTTP/1.1 vs /2, and session priming via the
app bundles. Static /Api/1/Blob assets fetch fine with the same session while every
contentapi call returns 200-with-nothing, so the token is the wall. A replayed token is
dead; only the page itself can mint a live one.

So we let the page do exactly that and read the responses it receives. No token
reverse-engineering, nothing to break when they redeploy their JS.

A persistent profile in .bet365_profile/ keeps cookies between runs, so later runs look
like a returning visitor rather than a fresh one each time.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fpl_pipeline import config  # noqa: E402
from tools.bet365 import parse_coupon, parse_wire  # noqa: E402

PROFILE = os.path.join(config.ROOT, ".bet365_profile")
FOOTBALL = "https://www.bet365.com/#/AC/B1/C1/D1002/E91422157/G40/"


def attach(port=9222):
    """Attach to YOUR OWN Chrome/Edge over CDP and record what it receives.

    Playwright's bundled browsers are patched builds and bet365 blocks them regardless of
    who is driving — the site stayed blank even under manual navigation. This connects to
    a real, unmodified browser you launched yourself, with your own profile and history,
    so there is nothing to detect. (CDP attach is Chromium-only; Firefox is not supported
    by Playwright, despite its --remote-debugging-port flag.)

    Launch it first, with every existing window of that browser CLOSED:
        chrome.exe --remote-debugging-port=9222
        msedge.exe --remote-debugging-port=9222
    """
    from playwright.sync_api import sync_playwright

    captured = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{port}")
        except Exception as exc:
            raise SystemExit(
                f"could not attach on port {port}: {type(exc).__name__}\n"
                "Close ALL windows of Chrome/Edge first, then relaunch it with\n"
                f'  chrome.exe --remote-debugging-port={port}\n'
                "and leave it open while this runs.")

        ctx = browser.contexts[0] if browser.contexts else browser.new_context()

        def on_response(resp):
            if "contentapi" not in resp.url:
                return
            try:
                body = resp.text()
            except Exception:
                return
            if not body:
                return
            captured.append((resp.url, body))
            if "coupon" in resp.url:
                name = next((r.get("NA") for r in parse_wire(body)
                             if r["_type"] == "EV" and " v " in (r.get("NA") or "")), "?")
                print(f"  captured coupon: {name} ({len(body)} chars)")

        ctx.on("response", on_response)
        print(f"attached to your browser on port {port} ({len(ctx.pages)} tabs open)")
        print("\n" + "=" * 70)
        print("BROWSE TO BET365 AND OPEN THE FIXTURES YOU WANT.")
        print("  Coupons appear below as they are recorded.")
        print("\nPress Enter here when you are done.")
        print("=" * 70)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
    return captured


def interactive(browser="firefox", start="https://www.bet365.com/"):
    """Open a browser, let YOU drive, and record every content response it receives.

    The reliable mode, and usually the right one. Automated navigation has to guess at
    their router, their volatile class names and their bot heuristics; a person clicking
    around has none of those problems. Every coupon you open is captured, so browsing the
    fixtures you care about IS the scrape.
    """
    from playwright.sync_api import sync_playwright

    captured = []
    with sync_playwright() as p:
        engine = getattr(p, browser)
        ctx = engine.launch_persistent_context(
            f"{PROFILE}_{browser}", headless=False, viewport={"width": 1500, "height": 950})
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")

        def on_response(resp):
            if "contentapi" not in resp.url:
                return
            try:
                body = resp.text()
            except Exception:
                return
            if not body:
                return
            captured.append((resp.url, body))
            if "coupon" in resp.url:
                name = next((r.get("NA") for r in parse_wire(body)
                             if r["_type"] == "EV" and " v " in (r.get("NA") or "")), "?")
                print(f"  captured coupon: {name} ({len(body)} chars)")

        ctx.on("response", on_response)          # context-level: covers new tabs too
        page.goto(start, wait_until="domcontentloaded")

        print("\n" + "=" * 70)
        print("BROWSE TO THE FIXTURES YOU WANT — every coupon you open is captured.")
        print("  Football > Premier League > click each match.")
        print("  Coupons appear below as they are recorded.")
        print("\nPress Enter here when you are done.")
        print("=" * 70)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass
        ctx.close()
    return captured


def collect(competition, limit=None, headed=False, timeout=45000, browser="firefox"):
    from playwright.sync_api import sync_playwright

    captured = []           # (url, body) for every contentapi response the page receives

    with sync_playwright() as p:
        # Firefox by default: Chromium was fingerprinted as automation here — after
        # navigating, the SPA issued NO content requests at all and rendered blank,
        # while the same journey in a real Firefox works.
        engine = getattr(p, browser)
        kwargs = {"headless": not headed, "viewport": {"width": 1500, "height": 950}}
        if browser == "chromium":
            kwargs["args"] = ["--disable-blink-features=AutomationControlled"]
        ctx = engine.launch_persistent_context(f"{PROFILE}_{browser}", **kwargs)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        # Remove the most obvious automation tell before any page script runs
        page.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")

        def on_response(resp):
            if "contentapi" not in resp.url:
                return
            try:
                body = resp.text()
            except Exception:
                return
            if body:
                captured.append((resp.url, body))

        page.on("response", on_response)

        # Land on the homepage first and let the SPA boot fully — a deep hash link can
        # be applied before the router is ready, which leaves it on the loading screen.
        print("opening bet365 ...")
        page.goto("https://www.bet365.com/", wait_until="domcontentloaded", timeout=timeout)
        for _ in range(20):                      # up to ~40s for the app to appear
            page.wait_for_timeout(2000)
            if page.locator("div.wn-Classification, div.sm-CouponLink, "
                            "div.hm-MainHeaderRHSLoggedOutWide").count():
                break
        print(f"  app booted, {len(captured)} contentapi responses so far")

        # Never hardcode ids: the ones in an old HAR go stale and their router answers
        # with notfoundcontentapi/notfound, leaving a blank page. The left-nav menu the
        # app just fetched carries the CURRENT ids, so read them from that.
        menu = [(r.get("NA", ""), r.get("PD", "")) for url, body in captured
                if "allsportsmenu" in url for r in parse_wire(body)
                if r.get("NA") and r.get("PD")]
        # The left nav lists sports and featured competitions only — individual leagues
        # sit beneath them, so fall back to the upcoming-football hub, which lists every
        # forthcoming match including the Premier League.
        target = None
        for want in (competition, "Upcoming Football", "Football"):
            target = next(((n, p) for n, p in menu if want.lower() in n.lower()), None)
            if target:
                if want != competition:
                    print(f"  '{competition}' not in the top-level menu; using '{target[0]}'")
                break
        if target:
            name, pd = target
            # Click the nav entry rather than goto-ing a hash URL. A hash change is a
            # same-document navigation, so the router may never run — which is what left
            # the page blank and fired no content requests at all.
            print(f"clicking '{name}' in the nav (live id {pd})")
            clicked = False
            for attempt in (lambda: page.get_by_text(name, exact=True).first,
                            lambda: page.locator(f"[data-nav*='{pd[:18]}']").first,
                            lambda: page.get_by_text(name.split()[0], exact=False).first):
                try:
                    attempt().click(timeout=8000)
                    clicked = True
                    break
                except Exception:
                    continue
            if not clicked:
                route = "https://www.bet365.com/#" + pd.replace("#", "/")
                print(f"  click failed, falling back to {route[:60]}")
                page.goto(route, wait_until="domcontentloaded", timeout=timeout)
        else:
            print("no football entry in the live menu — clicking through the UI")
            for label in ("Football", "Soccer"):
                try:
                    page.get_by_text(label, exact=True).first.click(timeout=6000)
                    break
                except Exception:
                    continue
        page.wait_for_timeout(12000)

        # bet365's class names are volatile, so try several and use whichever hits.
        candidates = [
            "div.rcl-ParticipantFixtureDetailsTeam_TeamName",
            "div.sl-CouponParticipantWithBookCloses_Name",
            "div.scb-ParticipantFixtureDetailsHigherHalf_Team",
            "div.ovm-FixtureDetailsTwoWay_TeamsWrapper",
            "[class*='ParticipantFixtureDetails']",
            "[class*='CouponParticipant']",
        ]
        print(f"looking for {competition} fixtures ...")
        links, count = None, 0
        for sel in candidates:
            n = page.locator(sel).count()
            print(f"  {n:>4}  {sel}")
            if n and not count:
                links, count = page.locator(sel), n
        if not count:
            print(f"  live menu holds {len(menu)} entries; football-related ones:")
            for nm, pd in menu:
                if any(w in nm.lower() for w in ("football", "premier", "league", "soccer")):
                    print(f"    {nm[:44]:<46} {pd[:44]}")
            shot = os.path.join(config.OUTPUTS_DIR, "bet365_page.png")
            page.screenshot(path=shot, full_page=True)
            html = os.path.join(config.OUTPUTS_DIR, "bet365_page.html")
            open(html, "w", encoding="utf-8").write(page.content())
            print(f"  nothing matched — saved {os.path.basename(shot)} and "
                  f"{os.path.basename(html)} to outputs/ for inspection")

        seen = 0
        for i in range(count):
            if limit and seen >= limit:
                break
            try:
                before = len(captured)
                links.nth(i).click(timeout=8000)
                page.wait_for_timeout(4000)      # wait for the coupon response
                if len(captured) > before:
                    seen += 1
                    print(f"  [{seen}] captured a coupon ({len(captured)} responses so far)")
                page.go_back(wait_until="domcontentloaded")
                page.wait_for_timeout(2500)
            except Exception as exc:
                print(f"  fixture {i}: {type(exc).__name__}")
                continue

        ctx.close()
    return captured


def to_rows(captured):
    rows = []
    for url, body in captured:
        if "matchbettingcontentapi/coupon" not in url:
            continue
        records = parse_wire(body)
        name = next((r.get("NA") for r in records
                     if r["_type"] == "EV" and " v " in (r.get("NA") or "")), None)
        if not name:
            name = next((r.get("NA") for r in records if r["_type"] == "EV" and r.get("NA")),
                        "unknown")
        rows += parse_coupon(records, name)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--competition", default="England Premier League")
    ap.add_argument("--limit", type=int, help="stop after this many fixtures")
    ap.add_argument("--headed", action="store_true", help="show the browser")
    ap.add_argument("--browser", default="firefox", choices=("firefox", "chromium", "webkit"),
                    help="firefox by default — chromium was fingerprinted as automation")
    ap.add_argument("--interactive", action="store_true",
                    help="you browse, we record (the reliable mode)")
    ap.add_argument("--attach", action="store_true",
                    help="attach to your OWN Chrome/Edge started with --remote-debugging-port")
    ap.add_argument("--port", type=int, default=9222)
    args = ap.parse_args()

    try:
        if args.attach:
            captured = attach(args.port)
        elif args.interactive:
            captured = interactive(browser=args.browser)
        else:
            captured = collect(args.competition, args.limit, args.headed, browser=args.browser)
    except ImportError:
        sys.exit("pip install playwright && playwright install chromium")

    print(f"\n{len(captured)} contentapi responses captured")
    for url, body in captured:
        print(f"  {url.split('bet365.com/')[-1].split('?')[0]:<44} {len(body):>7} chars")
    rows = to_rows(captured)
    if not rows:
        sys.exit("no coupons captured — rerun with --headed to see what the page did")

    df = pd.DataFrame(rows)
    out = os.path.join(config.OUTPUTS_DIR, "bet365_odds.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} selections across {df['fixture'].nunique()} fixtures, "
          f"{df['market_group'].nunique()} market groups -> {os.path.relpath(out, config.ROOT)}")
    print("\nfixtures:", ", ".join(sorted(df["fixture"].unique())))
