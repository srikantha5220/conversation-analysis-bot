from docx import Document
from pathlib import Path

p = Path('report.md')
if not p.exists():
    raise SystemExit('report.md not found')

text = p.read_text(encoding='utf-8')
lines = text.splitlines()

doc = Document()

in_code = False
for line in lines:
    stripped = line.strip()
    if stripped.startswith('```'):
        in_code = not in_code
        continue
    if in_code:
        doc.add_paragraph(line, style='Normal')
        continue
    if stripped.startswith('# '):
        doc.add_heading(stripped.lstrip('# ').strip(), level=1)
    elif stripped.startswith('## '):
        doc.add_heading(stripped.lstrip('# ').strip(), level=2)
    elif stripped.startswith('### '):
        doc.add_heading(stripped.lstrip('# ').strip(), level=3)
    elif not stripped:
        doc.add_paragraph('')
    else:
        doc.add_paragraph(line)

out = Path('report.docx')
doc.save(out)
print(f'Wrote {out.resolve()}')
