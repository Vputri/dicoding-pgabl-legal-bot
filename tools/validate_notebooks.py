"""Validasi notebook: JSON well-formed + setiap sel kode lolos parsing Python.

Baris yang diawali ! atau % (magic Colab) diabaikan saat parsing.
"""
import ast
import json
import glob
import pathlib
import sys

root = pathlib.Path(__file__).parent.parent
gagal = 0

for path in sorted(glob.glob(str(root / "*.ipynb"))):
    nb = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    name = pathlib.Path(path).name
    n_code = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        n_code += 1
        src = "".join(cell["source"])
        clean_lines, in_magic = [], False
        for l in src.split("\n"):
            if in_magic or l.lstrip().startswith(("!", "%")):
                in_magic = l.rstrip().endswith("\\")   # ikut buang baris lanjutan
                clean_lines.append("pass")
            else:
                clean_lines.append(l)
        clean = "\n".join(clean_lines)
        try:
            ast.parse(clean)
        except SyntaxError as e:
            gagal += 1
            print(f"[SYNTAX] {name} sel #{i}: {e}")
            print("   >>", (e.text or "").rstrip())
    print(f"[OK] {name}: {len(nb['cells'])} sel ({n_code} kode)")

# Cek token tidak pernah ditulis literal (larangan rubrik no. 9).
import re

TOKEN_RE = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})")
for path in sorted(glob.glob(str(root / "*.ipynb"))) + [str(root / "link_huggingface.txt")]:
    p = pathlib.Path(path)
    if not p.exists():
        continue
    hit = TOKEN_RE.search(p.read_text(encoding="utf-8"))
    if hit:
        gagal += 1
        print(f"[SECRET] {p.name}: ditemukan token literal {hit.group()[:10]}...")

print("\nHasil:", "SEMUA LOLOS" if gagal == 0 else f"{gagal} masalah")
sys.exit(1 if gagal else 0)
