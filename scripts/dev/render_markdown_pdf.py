from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from markdown import markdown


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 11pt;
      line-height: 1.5;
      color: #111;
      background: #fff;
    }}
    .page {{
      max-width: 800px;
      margin: 0 auto;
      padding: 0;
    }}
    h1, h2, h3, h4 {{
      line-height: 1.2;
      margin: 1.2em 0 0.4em;
    }}
    h1 {{ font-size: 24pt; }}
    h2 {{ font-size: 16pt; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
    h3 {{ font-size: 13pt; }}
    p, li {{
      orphans: 3;
      widows: 3;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0.8em 0 1em;
      font-size: 10pt;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 6px 8px;
      vertical-align: top;
    }}
    th {{
      background: #f3f4f6;
      text-align: left;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 0.8em auto;
    }}
    code {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 0.95em;
    }}
    pre {{
      background: #f8f8f8;
      border: 1px solid #e5e7eb;
      padding: 12px;
      overflow-x: auto;
    }}
    .markdown-body > :first-child {{
      margin-top: 0;
    }}
  </style>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
        processEscapes: true,
        tags: 'none'
      }},
      options: {{
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
  <div class="page markdown-body">
  {body}
  </div>
</body>
</html>
"""


def find_msedge() -> str:
    candidates = [
        shutil.which("msedge"),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Could not locate msedge.exe")


def build_html(markdown_path: Path, title: str) -> str:
    source = markdown_path.read_text(encoding="utf-8")
    body = markdown(
        source,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "sane_lists",
            "toc",
        ],
        output_format="html5",
    )
    return HTML_TEMPLATE.format(title=title, body=body)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a markdown report to PDF via Edge.")
    parser.add_argument("markdown", type=Path, help="Source markdown file")
    parser.add_argument("pdf", type=Path, help="Output PDF path")
    parser.add_argument("--title", default=None, help="Document title")
    parser.add_argument("--keep-html", action="store_true", help="Keep the generated HTML file")
    args = parser.parse_args()

    markdown_path = args.markdown.resolve()
    pdf_path = args.pdf.resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    title = args.title or markdown_path.stem.replace("_", " ").title()
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(build_html(markdown_path, title), encoding="utf-8")

    edge = find_msedge()
    cmd = [
        edge,
        "--headless",
        "--disable-gpu",
        "--no-first-run",
        "--allow-file-access-from-files",
        "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=30000",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)

    if not args.keep_html:
        html_path.unlink(missing_ok=True)

    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
