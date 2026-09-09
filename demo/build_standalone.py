#!/usr/bin/env python
"""Fold data.js into index.html to make one self-contained file.

    python demo/build_standalone.py

Produces demo/web/standalone.html, one file, no server, no sibling files, no
network beyond the webfont (which degrades to the fallback stack). Mail it,
put it on a USB stick, open it on any laptop in the room.
"""
from pathlib import Path

WEB = Path(__file__).parent / "web"


def main() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    data = (WEB / "data.js").read_text(encoding="utf-8")

    tag = '<script src="data.js"></script>'
    if tag not in html:
        raise SystemExit(f"expected {tag!r} in index.html")

    # A closing tag inside the JSON payload would end the script element early.
    # It cannot occur in base64 or in our keys, but check rather than assume.
    if "</script" in data:
        raise SystemExit("data contains a closing script tag; escape before inlining")

    out = WEB / "standalone.html"
    out.write_text(html.replace(tag, "<script>\n" + data + "\n</script>"), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
