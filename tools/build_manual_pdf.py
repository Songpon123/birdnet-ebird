"""สร้าง คู่มือการใช้.pdf จาก คู่มือการใช้.md (+ ภาพใน docs/images) ด้วย Google Chrome แบบ headless

ใช้:  python3 tools/build_manual_pdf.py            -> คู่มือการใช้.pdf ข้างไฟล์ .md
แปลง Markdown เฉพาะรูปแบบที่คู่มือใช้ (หัวข้อ ตาราง รายการซ้อน โค้ด ภาพ ข้อความเน้น) ไม่ต้องติดตั้งอะไรเพิ่ม
ลิงก์ไปไฟล์อื่นใน repo ชี้ไปที่ GitHub แทน path ในเครื่อง (PDF ส่งต่อได้โดยไม่มีชื่อผู้ใช้ติดไป)"""
import html
import re
import sys
import unicodedata
from pathlib import Path

import subprocess
import tempfile

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "คู่มือการใช้.md"
PDF = BASE / "คู่มือการใช้.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def slug(text):
    """แบบ GitHub: ตัวเล็ก ตัดเครื่องหมายวรรคตอน/สัญลักษณ์ (ยกเว้น -) ช่องว่าง -> -"""
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    kept = "".join(ch for ch in text if ch == "-" or not unicodedata.category(ch).startswith(("P", "S")))
    return kept.replace(" ", "-")


def inline(text):
    codes = []

    def keep(match):
        codes.append(f"<code>{html.escape(match.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"
    text = re.sub(r"`([^`]+)`", keep, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", lambda m: f'<img alt="{m.group(1)}" src="{(BASE / m.group(2)).as_uri()}">', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: f'<a href="{m.group(2) if m.group(2).startswith(("#", "http")) else 'https://github.com/Songpon123/birdnet-ebird/blob/main/' + m.group(2)}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], text)


def table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines]
    head, body = rows[0], rows[2:]
    out = ["<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body]
    return "\n".join(out + ["</tbody></table>"])


def lists(lines):
    """รายการซ้อนตามระยะย่อหน้า: - / 1. ; บรรทัดต่อที่ย่อหน้า = ข้อความต่อของข้อเดิม"""
    out, stack = [], []          # stack: [(indent, tag, ซ้อนอยู่ใน li ของข้อแม่)]
    for line in lines:
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if not m:
            if out and out[-1].endswith("</li>") and out[-1].startswith("<li>"):
                out[-1] = out[-1][:-5] + "<br>" + inline(line.strip()) + "</li>"
            else:
                out.append(f"<li class=cont>{inline(line.strip())}</li>")
            continue
        indent, marker, text = len(m.group(1)), m.group(2), m.group(3)
        tag = "ol" if marker[0].isdigit() else "ul"
        while stack and indent < stack[-1][0]:
            _indent, closing, nested = stack.pop()
            out.append(f"</{closing}>" + ("</li>" if nested else ""))
        if not stack or indent > stack[-1][0]:
            nested = bool(stack) and out[-1].endswith("</li>")
            if nested:
                out[-1] = out[-1][:-5]
            stack.append((indent, tag, nested))
            start = f' start="{marker[:-1]}"' if tag == "ol" and marker != "1." else ""
            out.append(f"<{tag}{start}>")
        out.append(f"<li>{inline(text)}</li>")
    while stack:
        _indent, closing, nested = stack.pop()
        out.append(f"</{closing}>" + ("</li>" if nested else ""))
    return "\n".join(out)


def convert(md):
    lines = md.split("\n")
    out, i, first_h1 = [], 0, True
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and first_h1:
                first_h1 = False
                i += 1
                continue                         # หน้าปกมีชื่อแล้ว
            out.append(f'<h{level} id="{slug(text)}"' + (' class="toc"' if text == 'สารบัญ' else '') + f'>{inline(text)}</h{level}>')
            i += 1
            continue
        if line.strip() == "---":
            i += 1
            continue
        if line.startswith(">"):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            quote = [l[1:].strip() for l in lines[i:j]]
            paras = "\n".join(quote).split("\n\n")
            out.append("<blockquote>" + "".join(f"<p>{inline(' '.join(p.split(chr(10))))}</p>" for p in paras if p.strip()) + "</blockquote>")
            i = j
            continue
        if line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            j = i
            while j < len(lines) and lines[j].strip() and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[j]) or lines[j].startswith(" ")):
                j += 1
            out.append(lists(lines[i:j]))
            i = j
            continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", line)
        if m:
            out.append(f'<figure><img alt="{html.escape(m.group(1))}" src="{(BASE / m.group(2)).as_uri()}"><figcaption>{inline(m.group(1))}</figcaption></figure>')
            i += 1
            continue
        if not line.strip():
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||>|```|!\[|\s*([-*]|\d+\.)\s)", lines[j]):
            j += 1
        out.append("<p>" + inline(" ".join(l.strip() for l in lines[i:j])) + "</p>")
        i = j
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 17mm 15mm 18mm 15mm;
  @top-right { content: "คู่มือการใช้ BirdNET → eBird clipper"; font: 8.5pt 'Sukhumvit Set', sans-serif; color: #8a8f98; }
  @bottom-center { content: counter(page) " / " counter(pages); font: 9pt 'Sukhumvit Set', sans-serif; color: #8a8f98; } }
@page :first { @top-right { content: none; } @bottom-center { content: none; } }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Sukhumvit Set', 'Thonburi', -apple-system, sans-serif; font-size: 10.5pt; line-height: 1.62; color: #1d2128; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.cover img { width: 30mm; margin-bottom: 10mm; }
.cover h1 { font-size: 30pt; line-height: 1.25; margin: 0 0 6mm; color: #12151c; border: 0; }
.cover .sub { font-size: 13pt; color: #4a5260; margin-bottom: 18mm; }
.cover .meta { font-size: 10pt; color: #6b7280; border-top: 1px solid #d8dce3; padding-top: 5mm; }
h2 { font-size: 17pt; color: #0f3d5e; border-bottom: 2px solid #2f7fb5; padding-bottom: 2mm; margin: 0 0 4mm; page-break-before: always; }
h3 { font-size: 13pt; color: #1f5f8b; margin: 6mm 0 2mm; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 4mm 0 1mm; page-break-after: avoid; }
p { margin: 1.6mm 0; }
a { color: #1f6fb0; text-decoration: none; }
code { font-family: 'SF Mono', Menlo, monospace; font-size: 8.8pt; background: #eef1f5; padding: 0.2mm 1mm; border-radius: 1mm; }
pre { font-family: 'SF Mono', Menlo, 'Sukhumvit Set', monospace; font-size: 8.3pt; line-height: 1.45; background: #f4f6f9;
      border: 1px solid #dde2e9; border-radius: 2mm; padding: 3mm 4mm; white-space: pre-wrap; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 2.5mm 0 3.5mm; font-size: 9.4pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background: #e8eef5; color: #0f3d5e; text-align: left; font-weight: 600; }
th, td { border: 1px solid #d3dae3; padding: 1.4mm 2.2mm; vertical-align: top; }
tbody tr:nth-child(even) td { background: #fafbfc; }
blockquote { margin: 3mm 0; padding: 2mm 4mm; background: #fff8e6; border-left: 3px solid #e0a526; border-radius: 1mm; page-break-inside: avoid; }
blockquote p { margin: 1mm 0; }
figure { margin: 3mm 0 4mm; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 205mm; border: 1px solid #c9d0da; border-radius: 1.5mm; }
figcaption { font-size: 8.8pt; color: #6b7280; margin-top: 1mm; }
ul, ol { margin: 1.5mm 0; padding-left: 6mm; }
li { margin: 0.6mm 0; }
h2.toc + ol { columns: 2; column-gap: 10mm; }
h2.toc { page-break-before: avoid; margin-top: 8mm; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")
    body = convert(md)
    import itertools
    numbers = itertools.count(1)
    short = {old_id: f"s{next(numbers)}" for old_id in re.findall(r'<h[1-4] id="([^"]+)"', body)}
    body = re.sub(r'(id|href)="#?([^"]+)"', lambda m: f'{m.group(1)}="{"#" if m.group(1) == "href" else ""}{short[m.group(2)]}"'
                  if m.group(2) in short else m.group(0), body)
    icon = (BASE / "app/assets/app_icon.png").as_uri()
    cover = f"""<section class="cover"><img src="{icon}" alt="">
<h1>คู่มือการใช้<br>BirdNET → eBird clipper</h1>
<div class="sub">ตัดเสียงนกจากไฟล์ภาคสนามด้วย BirdNET ฟังตรวจทีละคลิป แล้วเตรียมอัปโหลดขึ้น eBird / Macaulay Library</div>
<div class="meta">macOS / Windows · ฉบับพฤศจิกายน 2026 · ภาพหน้าจอจากข้อมูลตัวอย่าง</div></section>"""
    work = Path(tempfile.mkdtemp(prefix="manual-pdf-"))
    page = work / "manual.html"
    page.write_text(f"""<!doctype html><html lang="th"><head><meta charset="utf-8"><title>คู่มือการใช้ BirdNET → eBird clipper</title>
<style>{CSS}</style></head><body>{cover}
{body}</body></html>""", encoding="utf-8")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                    f"--user-data-dir={work / 'profile'}", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    "--virtual-time-budget=10000", f"--print-to-pdf={PDF}", page.as_uri()],
                   check=True, capture_output=True, timeout=300)
    print(f"{PDF} ({PDF.stat().st_size / 1e6:.1f} MB)")


main()
