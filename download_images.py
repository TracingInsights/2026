#!/usr/bin/env python3
"""Save official Formula 1 driver and car images into images/.

Three sets are collected, all keyed by the `team` and `driver` fields of the
per-session drivers.json files so a path can be built straight from any entry:

  images/drivers/<team>/<driver>.webp   full-body cutout used on
                                        https://www.formula1.com/en/drivers
  images/headshots/<team>/<driver>.png  840x840 headshot behind the `url`
                                        field of drivers.json (legacy asset)
  images/cars/<team>.webp               car side view used on
                                        https://www.formula1.com/en/teams

Driver + team pairing matters: a driver who stands in at another team (Liam
Lawson at Red Bull Racing while Isack Hadjar was out, with Yuki Tsunoda in
Lawson's Racing Bulls seat) gets one image per team.

Every file is the untransformed source asset served by media.formula1.com,
with no resizing or re-encoding. Headshots whose source asset the host no
longer has fall back to the largest rendition it still serves. The host answers
a missing asset with a generic placeholder and an `x-cld-error` header (or a
404); such responses are never saved and the pairing is reported instead.

images/manifest.json lists every saved file with pixel size and source URL.

  python3 download_images.py                 # fetch what is missing
  python3 download_images.py --force         # re-download everything
  python3 download_images.py --only cars     # one set: drivers | headshots | cars
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "images"
SEASON = 2026
CDN = f"https://media.formula1.com/image/upload/v1740000001/common/f1/{SEASON}"
USER_AGENT = "Mozilla/5.0 (TracingInsights image sync)"
SKIP_PREFIXES = (".", "cache", "_")
SETS = ("drivers", "headshots", "cars")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TRANSFORM_RE = re.compile(r"^(?P<asset>.+\.png)\.transform/[^/]+/image\.png$")
LEGACY_SLUG_RE = re.compile(r"/drivers/[A-Z]/[A-Z0-9]+_[^/]+/([a-z]{6}\d{2})\.png")
# Legacy host renditions, largest first. Used only when the source asset is missing.
RENDITIONS = ("12col", "9col", "4col-retina", "6col", "4col", "3col", "2col-retina", "2col", "1col")

# Legacy headshot asset for pairings whose drivers.json `url` is "None".
HEADSHOT_OVERRIDES: dict[tuple[str, str], str] = {
    ("TSU", "Racing Bulls"): (
        "https://media.formula1.com/d_driver_fallback_image.png/content/dam/"
        "fom-website/drivers/Y/YUKTSU01_Yuki_Tsunoda/yuktsu01.png"
    ),
}
# formula1.com driver slug when it cannot be read from the legacy url or
# guessed from the name (first 3 letters of first + last name + "01").
SLUG_OVERRIDES: dict[str, str] = {}


class Unavailable(Exception):
    """The host has no such asset (404 or its generic placeholder)."""


# --------------------------------------------------------------------------- data

def _real(value: object) -> str | None:
    return value if isinstance(value, str) and value and value != "None" else None


def collect_pairings(root: Path) -> dict[tuple[str, str], dict]:
    """Return {(driver, team): {"fn", "ln", "urls": [...]}} from every drivers.json."""
    pairings: dict[tuple[str, str], dict] = {}
    for path in sorted(root.glob("*/*/drivers.json")):
        if path.relative_to(root).parts[0].startswith(SKIP_PREFIXES):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {path.relative_to(root)}: {exc}", file=sys.stderr)
            continue
        for entry in data.get("drivers", []):
            code, team = _real(entry.get("driver")), _real(entry.get("team"))
            if not code or not team:
                continue
            info = pairings.setdefault((code, team), {"fn": None, "ln": None, "urls": []})
            info["fn"] = info["fn"] or _real(entry.get("fn"))
            info["ln"] = info["ln"] or _real(entry.get("ln"))
            url = _real(entry.get("url"))
            if url and url not in info["urls"]:
                info["urls"].append(url)
    return pairings


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def team_slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _ascii(team).lower())


def driver_slug(code: str, info: dict) -> str | None:
    for url in info["urls"]:
        m = LEGACY_SLUG_RE.search(url)
        if m:
            return m.group(1)
    if code in SLUG_OVERRIDES:
        return SLUG_OVERRIDES[code]
    fn, ln = info.get("fn"), info.get("ln")
    if fn and ln:
        return (_ascii(fn)[:3] + _ascii(ln)[:3]).lower() + "01"
    return None


# ---------------------------------------------------------------------- fetching

def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            if resp.headers.get("x-cld-error"):
                raise Unavailable("placeholder")
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise Unavailable("404") from exc
        raise
    if not (data.startswith(PNG_MAGIC) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")):
        raise ValueError("response is not a PNG or WebP")
    return data


def image_size(data: bytes) -> tuple[int, int] | None:
    if data.startswith(PNG_MAGIC):
        return struct.unpack(">II", data[16:24])
    chunk = data[12:16]
    if chunk == b"VP8X":
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return w, h
    if chunk == b"VP8L":
        b0, b1, b2, b3 = data[21:25]
        w = 1 + (b0 | (b1 & 0x3F) << 8)
        h = 1 + ((b1 >> 6) | b2 << 2 | (b3 & 0x0F) << 10)
        return w, h
    if chunk == b"VP8 ":
        w = int.from_bytes(data[26:28], "little") & 0x3FFF
        h = int.from_bytes(data[28:30], "little") & 0x3FFF
        return w, h
    return None


def headshot_candidates(url: str) -> list[tuple[str, str]]:
    """(rendition, url) pairs for a legacy headshot, best quality first."""
    m = TRANSFORM_RE.match(url)
    asset = m.group("asset") if m else url
    out = [("original", asset)]
    out += [(t, f"{asset}.transform/{t}/image.png") for t in RENDITIONS]
    if url not in {u for _, u in out}:
        out.append(("listed", url))
    return out


# ------------------------------------------------------------------------ runner

class Runner:
    def __init__(self, force: bool, previous: dict[str, dict]):
        self.force = force
        self.previous = previous  # file -> previous manifest entry
        self.cache: dict[str, bytes | Unavailable] = {}
        self.downloaded = self.kept = self.failed = 0
        self.unavailable: list[str] = []

    def acquire(self, label: str, dest: Path, candidates: list[tuple[str, str]], extra: dict) -> dict | None:
        rel = dest.relative_to(IMAGES).as_posix()
        if dest.exists() and not self.force:
            self.kept += 1
            data = dest.read_bytes()
            prev = self.previous.get(rel, {})
            rendition, source = prev.get("rendition"), prev.get("source")
        else:
            data, rendition, source = b"", None, None
            missing = True
            for name, url in candidates:
                got = self.cache.get(url)
                if got is None:
                    try:
                        got = fetch(url)
                    except Unavailable as exc:
                        got = exc
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {label}: {url} -> {exc}", file=sys.stderr)
                        missing = False
                        continue
                    self.cache[url] = got
                if isinstance(got, Unavailable):
                    continue
                data, rendition, source = got, name, url
                break
            if not data:
                if missing:
                    self.unavailable.append(label)
                else:
                    self.failed += 1
                    print(f"FAILED {label}", file=sys.stderr)
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            self.downloaded += 1
            size = image_size(data)
            dims = f"{size[0]}x{size[1]}" if size else "?"
            print(f"saved images/{rel} ({dims}, {rendition}, {len(data)} bytes)")

        size = image_size(data)
        entry = {**extra, "file": rel}
        if size:
            entry["width"], entry["height"] = size
        entry["rendition"] = rendition
        entry["source"] = source
        return entry


def load_previous(path: Path) -> dict[str, dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[str, dict] = {}
    for entries in doc.values() if isinstance(doc, dict) else []:
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict) and e.get("file"):
                out[e["file"]] = e
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=ROOT, help="repo root to scan for drivers.json")
    p.add_argument("--force", action="store_true", help="re-download images that already exist")
    p.add_argument("--only", choices=SETS, action="append", help="limit to one or more sets")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sets = tuple(args.only) if args.only else SETS
    pairings = collect_pairings(args.root)
    if not pairings:
        print("no driver/team pairings found", file=sys.stderr)
        return 1

    IMAGES.mkdir(parents=True, exist_ok=True)
    manifest_path = IMAGES / "manifest.json"
    previous = load_previous(manifest_path)
    run = Runner(args.force, previous)
    manifest: dict[str, list[dict]] = {s: [] for s in SETS}
    # keep entries for sets not refreshed this run
    for e in previous.values():
        top = e["file"].split("/", 1)[0]
        if top in SETS and top not in sets:
            manifest[top].append(e)

    for (code, team), info in sorted(pairings.items()):
        ident = {"driver": code, "team": team, "fn": info["fn"], "ln": info["ln"]}
        label = f"{code} @ {team}"

        if "drivers" in sets:
            slug = driver_slug(code, info)
            if slug:
                tslug = team_slug(team)
                url = f"{CDN}/{tslug}/{slug}/{SEASON}{tslug}{slug}right.webp"
                entry = run.acquire(f"driver {label}", IMAGES / "drivers" / team / f"{code}.webp", [("original", url)], ident)
                if entry:
                    manifest["drivers"].append(entry)
            else:
                run.unavailable.append(f"driver {label} (no slug)")

        if "headshots" in sets:
            urls = info["urls"]
            if len(urls) > 1:
                print(f"warning: {label} lists {len(urls)} headshot urls, using the first", file=sys.stderr)
            start = HEADSHOT_OVERRIDES.get((code, team)) or (urls[0] if urls else None)
            if start:
                entry = run.acquire(f"headshot {label}", IMAGES / "headshots" / team / f"{code}.png", headshot_candidates(start), ident)
                if entry:
                    manifest["headshots"].append(entry)
            else:
                run.unavailable.append(f"headshot {label} (no url)")

    if "cars" in sets:
        for team in sorted({team for _, team in pairings}):
            tslug = team_slug(team)
            url = f"{CDN}/{tslug}/{SEASON}{tslug}carright.webp"
            entry = run.acquire(f"car {team}", IMAGES / "cars" / f"{team}.webp", [("original", url)], {"team": team})
            if entry:
                manifest["cars"].append(entry)

    for s in SETS:
        manifest[s].sort(key=lambda e: e["file"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if run.unavailable:
        print("not available on the host (not saved):")
        for item in run.unavailable:
            print(f"  {item}")
    print(
        f"done: {run.downloaded} downloaded, {run.kept} already present, {run.failed} failed, "
        f"manifest -> images/manifest.json"
    )
    return 1 if run.failed else 0


if __name__ == "__main__":
    sys.exit(main())
