"""Susun berkas submission menjadi PGABL_<nama>.zip sesuai ketentuan berkas.

Menolak membuat zip bila ada syarat yang belum terpenuhi (notebook belum dijalankan,
link_huggingface.txt masih placeholder, dsb.) agar tidak tersubmit dalam kondisi ditolak.
"""
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbutil import STUDENT  # noqa: E402

root = pathlib.Path(__file__).parent.parent
zip_path = root / f"PGABL_{STUDENT}.zip"

WAJIB = [
    f"Fine-tuning_submission_PGABL_{STUDENT}.ipynb",
    f"GRPO_submission_PGABL_{STUDENT}.ipynb",
    f"RAG_submission_PGABL_{STUDENT}.ipynb",
    "link_huggingface.txt",
    "requirements.txt",
]

masalah = []

for nama in WAJIB:
    p = root / nama
    if not p.exists():
        masalah.append(f"berkas hilang: {nama}")
        continue
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text(encoding="utf-8"))
        kode = [c for c in nb["cells"] if c["cell_type"] == "code"]
        kosong = [c for c in kode if not c.get("outputs")]
        if kosong:
            masalah.append(
                f"{nama}: {len(kosong)}/{len(kode)} sel kode belum punya output "
                "— jalankan notebook di Colab lalu unduh ulang"
            )

link = root / "link_huggingface.txt"
if link.exists():
    isi = link.read_text(encoding="utf-8")
    if "<username>" in isi or "BELUM DIISI" in isi:
        masalah.append("link_huggingface.txt masih berisi placeholder")

if masalah:
    print("Zip TIDAK dibuat. Perbaiki dulu:")
    for m in masalah:
        print("  -", m)
    raise SystemExit(1)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for nama in WAJIB:
        z.write(root / nama, f"PGABL_{STUDENT}/{nama}")

print(f"Zip dibuat: {zip_path.name} ({zip_path.stat().st_size / 1e6:.2f} MB)")
for n in zipfile.ZipFile(zip_path).namelist():
    print("  ", n)
