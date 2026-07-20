"""Uji reward function GRPO langsung dari sel notebook (tanpa perlu GPU/deps berat).

Fungsi yang diuji: format_reward_func & reasoning_length_reward — keduanya murni
string-processing, sehingga dapat diverifikasi lokal terhadap spesifikasi poin rubrik.
"""
import json
import pathlib

NB = next(pathlib.Path(__file__).parent.parent.glob("GRPO_submission_PGABL_*.ipynb"))
cells = json.loads(NB.read_text(encoding="utf-8"))["cells"]

ns = {}
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if any(k in src for k in ("def _reasoning_and_answer", "def format_reward_func",
                              "def reasoning_length_reward")):
        exec(compile(src, str(NB), "exec"), ns)

fmt = ns["format_reward_func"]
rlen = ns["reasoning_length_reward"]

def comp(text):
    return [[{"role": "assistant", "content": text}]]

LONG = "A" * 250
MID = "B" * 120
SHORT = "C" * 20

CASES = [
    # (label, teks, format_expected, reasoning_expected)
    ("format sempurna, reasoning panjang",
     f"<think>{LONG}</think> Jawaban final.", 1.0, 1.0),
    ("format sempurna, reasoning sedang",
     f"<think>{MID}</think> Jawaban final.", 1.0, 0.5),
    ("format sempurna, reasoning pendek",
     f"<think>{SHORT}</think> Jawaban final.", 1.0, 0.2),
    ("think kosong (hanya spasi)",
     "<think>   </think> Jawaban final.", 1.0, 0.0),
    ("tag ganda -> penalti",
     f"<think>{MID}</think> A <think>{MID}</think> B", -0.5, 0.5),
    ("penutup ganda -> penalti",
     f"<think>{MID}</think></think> A", -0.5, 0.5),
    ("terpotong: buka tanpa tutup",
     f"<think>{LONG}", 0.2, 1.0),
    ("tutup tanpa buka",
     f"{MID}</think> Jawaban.", 0.3, 0.0),
    ("tag tidak di awal kalimat",
     f"Menurut saya <think>{LONG}</think> Jawaban.", 0.5, 1.0),
    ("ditutup tapi tanpa jawaban akhir",
     f"<think>{LONG}</think>   ", 0.5, 1.0),
    ("tanpa tag sama sekali",
     "Jawaban langsung tanpa penalaran.", 0.0, 0.0),
]

gagal = 0
print(f"{'kasus':<38} {'format':>8} {'harap':>7} {'reason':>8} {'harap':>7}  status")
print("-" * 84)
for label, text, exp_fmt, exp_len in CASES:
    got_fmt = fmt(comp(text))[0]
    got_len = rlen(comp(text))[0]
    ok = abs(got_fmt - exp_fmt) < 1e-9 and abs(got_len - exp_len) < 1e-9
    gagal += not ok
    print(f"{label:<38} {got_fmt:>8.1f} {exp_fmt:>7.1f} {got_len:>8.1f} {exp_len:>7.1f}  "
          f"{'OK' if ok else 'GAGAL'}")

# Batas-batas persis yang disebut rubrik untuk reasoning_length_reward.
BOUNDARIES = [(49, 0.2), (50, 0.5), (199, 0.5), (200, 1.0)]
print("\nUji batas panjang reasoning:")
for n, expected in BOUNDARIES:
    got = rlen(comp("<think>" + "x" * n + "</think> jawab"))[0]
    ok = abs(got - expected) < 1e-9
    gagal += not ok
    print(f"  {n:>3} karakter -> {got:.1f} (harap {expected:.1f})  {'OK' if ok else 'GAGAL'}")

print("\nHasil:", "SEMUA LOLOS" if gagal == 0 else f"{gagal} kasus gagal")
raise SystemExit(1 if gagal else 0)
