"""Build the A12 rating page from the frozen inputs (claims/PREREG_A12_human_validation.md).

The rubric is judge_medicalization.USER verbatim (the part before QUESTION), the guide is
human_eval/rater_guide.md, the items are human_eval/items.json. Nothing on the page names a method.

    python human_eval/build_page.py        # writes human_eval/rating_page.html
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import judge_medicalization as jm   # noqa: E402

HERE = Path(__file__).resolve().parent


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def md_to_html(md: str) -> str:
    """Just enough Markdown for the guide: headings, paragraphs, bullet lists, one table."""
    out, para, lst, tbl = [], [], [], []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()
        if lst:
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in lst) + "</ul>")
            lst.clear()
        if tbl:
            head, rows = tbl[0], tbl[2:]
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]  # noqa: E731
            out.append('<div class="tablewrap"><table><thead><tr>' + "".join(f"<th>{inline(c)}</th>" for c in cells(head))
                       + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells(r))
                                                          + "</tr>" for r in rows) + "</tbody></table></div>")
            tbl.clear()
    for line in md.splitlines():
        if line.startswith("# "):
            flush()
            continue                                   # the page has its own title
        if line.startswith("## "):
            flush()
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("|"):
            if para or lst:
                flush()
            tbl.append(line)
        elif line.startswith("- "):
            if para or tbl:
                flush()
            lst.append(line[2:])
        elif line.startswith("  ") and lst:
            lst[-1] += " " + line.strip()
        elif not line.strip():
            flush()
        else:
            if lst or tbl:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def main() -> int:
    data = json.loads((HERE / "items.json").read_text())
    items_sha = hashlib.sha256((HERE / "items.json").read_bytes()).hexdigest()
    rubric = jm.USER.split("QUESTION:")[0].strip()
    guide = md_to_html((HERE / "rater_guide.md").read_text())
    tpl = (HERE / "rating_page.template.html").read_text()
    page = (tpl.replace("{{GUIDE}}", guide)
               .replace("{{RUBRIC}}", html.escape(rubric))
               .replace("{{DATA}}", json.dumps({"sha": items_sha[:12], **data}).replace("</", "<\\/")))
    (HERE / "rating_page.html").write_text(page)
    print(f"wrote {HERE / 'rating_page.html'} ({len(page) // 1024} KB), items sha {items_sha[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
