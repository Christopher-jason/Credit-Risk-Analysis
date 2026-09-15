#!/usr/bin/env python3
"""
Builds Credit_Risk_PD_Model_Business_Summary.pdf from report_source.html.

Edit the text in report_source.html, then run:

    python3 build_report.py

Only the standard library is needed - no pip installs. Rendering is done by
headless Chrome, which is what gives the PDF its print CSS (page breaks,
margins, table headers).

Why the inlining step: Chrome sandboxes local file:// pages and will silently
drop <img> tags that point outside the page's own directory. So before
rendering, every ../Pics/*.png reference is read off disk and embedded into a
temporary copy of the HTML as a base64 data URI. report_source.html itself
keeps the short readable paths.
"""

import base64
import mimetypes
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "report_source.html"
OUTPUT = HERE / "Credit_Risk_PD_Model_Business_Summary.pdf"

# First match wins. Add your own path here if Chrome lives somewhere unusual.
BROWSERS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
]


def find_browser():
    for path in BROWSERS:
        if path and pathlib.Path(path).exists():
            return path
    sys.exit(
        "No Chrome/Chromium found. Install Google Chrome, or add its path to "
        "the BROWSERS list at the top of this script."
    )


def inline_images(html):
    """Replace src="../Pics/foo.png" with an embedded base64 data URI."""
    missing = []

    def repl(match):
        quote, raw_src = match.group(1), match.group(2)
        if raw_src.startswith("data:"):
            return match.group(0)
        # Filenames may be URL-encoded in the HTML (spaces as %20).
        src = urllib.parse.unquote(raw_src)
        path = (HERE / src).resolve()
        if not path.exists():
            missing.append(src)
            return match.group(0)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        return f'src={quote}data:{mime};base64,{data}{quote}'

    html = re.sub(r'src=(["\'])(.*?)\1', repl, html)
    if missing:
        print("  WARNING - image(s) not found, will render blank:")
        for m in missing:
            print(f"    {m}")
    return html


def render(browser, html_path, pdf_path):
    """Run headless Chrome, then stop it once the PDF has been written.

    Chrome often keeps its process alive after --print-to-pdf finishes, so
    rather than waiting on it we poll for the output file and then terminate.
    """
    if pdf_path.exists():
        pdf_path.unlink()

    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(120):  # up to ~60s
                if pdf_path.exists() and pdf_path.stat().st_size > 0:
                    time.sleep(2)  # let the final flush complete
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        sys.exit("Render failed - no PDF was produced.")


def main():
    if not SOURCE.exists():
        sys.exit(f"Missing source file: {SOURCE}")

    browser = find_browser()
    print(f"Browser : {browser}")
    print(f"Source  : {SOURCE.name}")

    html = inline_images(SOURCE.read_text(encoding="utf-8"))

    # Chrome must read the inlined copy from beside the source so that any
    # relative paths still resolve the same way.
    tmp = HERE / ".report_inlined.tmp.html"
    tmp.write_text(html, encoding="utf-8")
    try:
        render(browser, tmp, OUTPUT)
    finally:
        tmp.unlink(missing_ok=True)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Output  : {OUTPUT.name}  ({size_kb:,.0f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
