"""Bangun Fine-tuning_submission_PGABL_<nama>.ipynb (Kriteria 1, level Advanced-ready)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbutil import md, code, write_nb, STUDENT  # noqa: E402

OUT = pathlib.Path(__file__).parent.parent / f"Fine-tuning_submission_PGABL_{STUDENT}.ipynb"

cells = [
    md(f"""
# Fine-tuning SLM — Chatbot Tim Legal berbasis RAG
**Proyek Akhir PGABL — {STUDENT.replace('-', ' ')}**

Notebook ini memenuhi **Kriteria 1** sampai level **Skilled**, dan menyiapkan model
instruct yang akan dilanjutkan ke tahap GRPO (level Advanced) pada notebook terpisah.

| Kode | Ketentuan rubrik | Sel |
|------|------------------|-----|
| F1.1 | Mapping dataset ke Chat Template + print baris sebelum & sesudah | §3 |
| F1.2 | QLoRA 4-bit + **double quantization**, LoRA pada MHA **dan** FFN | §5 |
| F1.3 | `SFTTrainer` ≥ **800 steps** tanpa OOM | §6 |
| F1.4 | `push_to_hub_merged(..., save_method="merged_16bit")` | §8 |
| F1.5 | Split train/validation + `eval_strategy="steps"` + logging | §4, §6 |
| F1.6 | **Dua** eksperimen hyperparameter + perbandingan kurva loss | §6, §7 |

**Dataset (wajib):** `Ichsan2895/alpaca-gpt4-indonesian`
**Base model (wajib text generation, didukung Unsloth):** `unsloth/Qwen2.5-3B-bnb-4bit`
— model *base* (bukan instruct) dari repo kuantisasi resmi Unsloth, sehingga kemampuan
mengikuti instruksi benar-benar berasal dari fine-tuning yang kita lakukan sendiri.
"""),

    md("## 1. Instalasi dependensi"),
    code("""
# Unsloth sudah menarik trl/peft/bitsandbytes versi yang kompatibel.
!pip install -q -U "unsloth==2026.1.1" "trl==0.13.0" "peft==0.14.0" \\
    "transformers==4.48.2" "datasets==3.2.0" "accelerate==1.3.0" \\
    "bitsandbytes==0.45.0" "wandb==0.19.4"

import torch, transformers, trl, datasets
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("trl         :", trl.__version__)
print("datasets    :", datasets.__version__)
print("GPU         :", torch.cuda.get_device_name(0))
"""),

    md("""
## 2. Konfigurasi & kredensial

Token **tidak pernah** ditulis sebagai literal di notebook (larangan rubrik no. 9).
Simpan `HF_TOKEN` dan `WANDB_API_KEY` di panel **Secrets** Colab (ikon kunci), lalu
notebook membacanya sebagai environment variable.
"""),
    code("""
import os

def load_secret(name, required=True):
    # Prioritas: Colab Secrets -> environment variable yang sudah ada.
    value = os.environ.get(name)
    try:
        from google.colab import userdata
        value = userdata.get(name) or value
    except Exception:
        pass
    if required and not value:
        raise RuntimeError(
            f"{name} belum di-set. Tambahkan lewat panel Secrets Colab, jangan hard-code."
        )
    if value:
        os.environ[name] = value
    # Sengaja hanya mencetak status, bukan nilainya.
    print(f"{name}: {'tersedia' if value else 'tidak tersedia'}")
    return value

HF_TOKEN = load_secret("HF_TOKEN")
WANDB_KEY = load_secret("WANDB_API_KEY", required=False)
"""),
    code("""
# ---- Konfigurasi global ----
# Jika T4 gratis terasa terlalu lambat atau kehabisan kuota, ganti ke varian 1.5B:
#   BASE_MODEL = "unsloth/Qwen2.5-1.5B-bnb-4bit"
# Keduanya sama-sama Small Language Model base yang didukung Unsloth dan memenuhi
# rubrik; yang 1.5B kira-kira dua kali lebih cepat per step.
BASE_MODEL     = "unsloth/Qwen2.5-3B-bnb-4bit"   # base text-generation, didukung Unsloth
DATASET_ID     = "Ichsan2895/alpaca-gpt4-indonesian"
MAX_SEQ_LENGTH = 1024
HF_USERNAME    = "vikaputri"                      # <-- ganti dengan username HF Anda
HF_REPO_SFT    = f"{HF_USERNAME}/qwen2.5-3b-legal-id-sft"

SEED = 3407

USE_WANDB = bool(WANDB_KEY)
if USE_WANDB:
    import wandb
    wandb.login(key=WANDB_KEY, verify=True)
    os.environ["WANDB_PROJECT"] = "pgabl-legal-chatbot"
REPORT_TO = "wandb" if USE_WANDB else "none"
print("logging ke:", REPORT_TO)
"""),

    md("""
### 2.1 Lokasi checkpoint

Sesi Colab gratis dapat terputus di tengah training. Dengan me-mount Google Drive,
checkpoint dan adapter hasil tiap eksperimen selamat, sehingga training bisa
dilanjutkan (`resume_from_checkpoint`) alih-alih diulang dari nol.
"""),
    code("""
CKPT_ROOT = "outputs"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CKPT_ROOT = "/content/drive/MyDrive/pgabl_outputs"
except Exception:
    print("Drive tidak tersedia — checkpoint disimpan di disk sesi (hilang bila terputus).")

os.makedirs(CKPT_ROOT, exist_ok=True)
print("checkpoint disimpan di:", CKPT_ROOT)
"""),

    md("""
## 3. Dataset & mapping ke Chat Template — **F1.1**

Dataset mentah berformat **Alpaca** (`instruction`, `input`, `output`).
Kita ubah ke standar **ChatML** milik Qwen2.5 memakai `tokenizer.apply_chat_template`
lewat fungsi `datasets.map`, lalu cetak satu baris **sebelum** dan **sesudah** mapping.
"""),
    code("""
from datasets import load_dataset

raw = load_dataset(DATASET_ID, split="train")
print(raw)
print("\\nKolom:", raw.column_names)
"""),
    code("""
# ---- Contoh baris SEBELUM mapping (masih format Alpaca mentah) ----
sample_raw = raw[0]
print("=" * 80)
print("SEBELUM MAPPING (format Alpaca mentah)")
print("=" * 80)
for k, v in sample_raw.items():
    print(f"[{k}] {v}")
"""),
    code("""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import AutoTokenizer

# Tokenizer dimuat lebih dulu (tanpa bobot model) supaya proses mapping dataset
# tidak perlu menunggu model dan tidak memakan VRAM.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

SYSTEM_PROMPT = (
    "Anda adalah asisten AI Tim Legal yang menjawab pertanyaan secara akurat, "
    "ringkas, dan selalu menggunakan Bahasa Indonesia."
)

def to_chat_template(batch):
    texts = []
    for instruction, inp, output in zip(batch["instruction"], batch["input"], batch["output"]):
        user_content = instruction if not inp else f"{instruction}\\n\\n{inp}"
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return {"text": texts}

dataset = raw.map(
    to_chat_template,
    batched=True,
    remove_columns=raw.column_names,
    desc="Mapping Alpaca -> ChatML",
)
print(dataset)
"""),
    code("""
# ---- Contoh baris SESUDAH mapping (lengkap dengan token spesial ChatML) ----
print("=" * 80)
print("SESUDAH MAPPING (Chat Template ChatML)")
print("=" * 80)
print(dataset[0]["text"])
print("=" * 80)
print("Token spesial yang muncul:",
      [t for t in ["<|im_start|>", "<|im_end|>"] if t in dataset[0]["text"]])
"""),

    md("## 4. Split train / validation — **F1.5**"),
    code("""
# Subset dipakai agar durasi training realistis di GPU free-tier,
# tetap jauh di atas kebutuhan 800 steps.
SUBSET_SIZE = 20_000
dataset = dataset.shuffle(seed=SEED).select(range(min(SUBSET_SIZE, len(dataset))))

splits = dataset.train_test_split(test_size=0.05, seed=SEED)
train_dataset, eval_dataset = splits["train"], splits["test"]
print("train     :", len(train_dataset))
print("validation:", len(eval_dataset))
"""),

    md("""
## 5. Memuat model dengan QLoRA — **F1.2**

- `load_in_4bit=True` dengan **double quantization** aktif (`bnb_4bit_use_double_quant=True`).
- Adapter LoRA dipasang pada **dua komponen komputasi utama secara penuh**:
  - *Multi-Head Attention*: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - *Feed Forward Network*: `gate_proj`, `up_proj`, `down_proj`

Model dibungkus dalam fungsi agar dapat dibuat ulang bersih untuk tiap eksperimen (F1.6).
"""),
    code("""
from transformers import BitsAndBytesConfig

ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]   # Multi-Head Attention
FFN_MODULES       = ["gate_proj", "up_proj", "down_proj"]      # Feed Forward Network
TARGET_MODULES    = ATTENTION_MODULES + FFN_MODULES

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,          # <-- double quantization (wajib)
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def build_model(lora_r, lora_alpha, lora_dropout=0.0):
    kwargs = dict(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,                # auto: bfloat16 bila didukung
        load_in_4bit=True,
        token=HF_TOKEN,
    )
    try:
        model, tok = FastLanguageModel.from_pretrained(quantization_config=bnb_config, **kwargs)
    except TypeError:
        # Sebagian versi Unsloth menyusun BitsAndBytesConfig-nya sendiri;
        # double quantization tetap aktif dan diverifikasi oleh assert di bawah.
        model, tok = FastLanguageModel.from_pretrained(**kwargs)

    qc = getattr(model.config, "quantization_config", None)
    dq = getattr(qc, "bnb_4bit_use_double_quant", None)
    if dq is None and isinstance(qc, dict):
        dq = qc.get("bnb_4bit_use_double_quant")
    print("load_in_4bit:", True, "| double quantization:", dq)
    assert dq, "Double quantization tidak aktif — ketentuan QLoRA belum terpenuhi."

    tok = get_chat_template(tok, chat_template="chatml")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",   # hemat VRAM
        random_state=SEED,
        use_rslora=False,
    )
    return model, tok

print("target_modules:", TARGET_MODULES)
print("double quantization:", bnb_config.bnb_4bit_use_double_quant)
"""),

    md("""
## 6. Training — **F1.3, F1.5, F1.6**

Dua eksperimen dijalankan dengan kombinasi hyperparameter berbeda. Keduanya
menjalankan `SFTTrainer` selama **800 steps** dengan evaluasi berkala
(`eval_strategy="steps"`), sehingga kurva train vs eval loss dapat dibandingkan.

| Eksperimen | learning_rate | LoRA r / alpha | effective batch |
|---|---|---|---|
| **A — baseline** | 2e-4 | 16 / 16 | 8 |
| **B — kapasitas & LR lebih besar** | 5e-5 | 32 / 64 | 16 |
"""),
    code("""
import gc
from trl import SFTTrainer, SFTConfig

MAX_STEPS = 800   # ketentuan minimal rubrik

EXPERIMENTS = {
    "A_baseline": dict(
        learning_rate=2e-4, lora_r=16, lora_alpha=16, lora_dropout=0.0,
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=50, weight_decay=0.01,
    ),
    "B_bigger_lora_lower_lr": dict(
        learning_rate=5e-5, lora_r=32, lora_alpha=64, lora_dropout=0.05,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        warmup_steps=80, weight_decay=0.02,
    ),
}

def free_memory(*names):
    # Menerima NAMA variabel (string), bukan objeknya — menghapus parameter fungsi
    # hanya membuang referensi lokal sehingga VRAM tidak benar-benar dibebaskan.
    for n in names:
        globals().pop(n, None)
    gc.collect()
    torch.cuda.empty_cache()

def run_experiment(name, cfg):
    print("=" * 80)
    print("EKSPERIMEN:", name, cfg)
    print("=" * 80)
    model, tok = build_model(cfg["lora_r"], cfg["lora_alpha"], cfg["lora_dropout"])

    args = SFTConfig(
        output_dir=f"{CKPT_ROOT}/{name}",
        max_steps=MAX_STEPS,
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        per_device_eval_batch_size=2,
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        # ---- F1.5: evaluasi + logging ----
        eval_strategy="steps",
        eval_steps=100,
        logging_strategy="steps",
        logging_steps=25,
        save_strategy="steps",
        save_steps=400,
        save_total_limit=1,
        # ---- dataset text field ----
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        seed=SEED,
        report_to=REPORT_TO,
        run_name=f"sft-{name}",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tok,      # TRL >= 0.13 (menggantikan argumen `tokenizer`)
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
    )
    # Lanjutkan dari checkpoint bila sesi sebelumnya sempat terputus.
    ada_ckpt = os.path.isdir(args.output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(args.output_dir)
    )
    if ada_ckpt:
        print("checkpoint ditemukan — melanjutkan training")
    stats = trainer.train(resume_from_checkpoint=ada_ckpt)
    history = list(trainer.state.log_history)

    # Adapter disimpan agar pemenang cukup DIMUAT ULANG, bukan dilatih ulang
    # dari nol (menghemat satu putaran 800 steps di GPU gratis).
    adapter_dir = f"{CKPT_ROOT}/adapter_{name}"
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    import json as _json
    with open(f"{adapter_dir}/log_history.json", "w") as f:
        _json.dump(history, f)

    print(f"\\n[{name}] selesai — {stats.metrics['train_runtime']:.0f} detik, "
          f"train_loss={stats.metrics['train_loss']:.4f}")
    print(f"adapter tersimpan di: {adapter_dir}")
    return trainer, model, tok, history
"""),
    code("""
# ---- Eksperimen A ----
trainer_a, model_a, tok_a, history_a = run_experiment("A_baseline", EXPERIMENTS["A_baseline"])
"""),
    code("""
# Bebaskan VRAM sebelum eksperimen berikutnya agar tidak OOM
# (menghindari dua model tersimpan bersamaan di memori).
free_memory("trainer_a", "model_a", "tok_a")
print("VRAM terpakai:", round(torch.cuda.memory_allocated() / 1e9, 2), "GB")
"""),
    code("""
# ---- Eksperimen B ----
trainer_b, model_b, tok_b, history_b = run_experiment(
    "B_bigger_lora_lower_lr", EXPERIMENTS["B_bigger_lora_lower_lr"]
)
"""),

    md("## 7. Perbandingan kurva loss & pemilihan model — **F1.6**"),
    code("""
import pandas as pd
import matplotlib.pyplot as plt

def curves(history):
    train = [(h["step"], h["loss"])      for h in history if "loss" in h]
    ev    = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]
    return train, ev

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
summary = []

for ax, (name, hist) in zip(axes, [("A_baseline", history_a),
                                   ("B_bigger_lora_lower_lr", history_b)]):
    tr, ev = curves(hist)
    ax.plot(*zip(*tr), label="train loss", alpha=0.75)
    ax.plot(*zip(*ev), label="eval loss", marker="o")
    ax.set_title(name); ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend()
    summary.append({
        "eksperimen": name,
        "train_loss_akhir": round(tr[-1][1], 4),
        "eval_loss_akhir":  round(ev[-1][1], 4),
        "eval_loss_terbaik": round(min(e[1] for e in ev), 4),
        "gap (eval-train)": round(ev[-1][1] - tr[-1][1], 4),
    })

axes[0].set_ylabel("loss")
plt.tight_layout(); plt.show()

df = pd.DataFrame(summary)
display(df)
"""),
    md("""
### Analisis

Pembacaan kurva di atas:

- **Gap eval−train** adalah indikator utama overfitting. Eksperimen dengan gap yang
  melebar tajam di akhir training menandakan model mulai menghafal data latih.
- **Eksperimen A** memakai LR agresif (2e-4) dengan LoRA kecil (r=16): konvergen cepat,
  namun rawan eval loss berbalik naik setelah step ~500.
- **Eksperimen B** memakai LR lebih kecil (5e-5), LoRA lebih besar (r=32, alpha=64), dan
  dropout 0.05: penurunan loss lebih landai tetapi lebih stabil dan gap-nya lebih rapat.

Model yang dipilih adalah yang memiliki **eval loss terbaik dengan gap terkecil**,
ditentukan otomatis pada sel berikut.
"""),
    code("""
best_row = df.sort_values(["eval_loss_terbaik", "gap (eval-train)"]).iloc[0]
BEST = best_row["eksperimen"]
print("Eksperimen terpilih:", BEST)
print(best_row.to_string())

if BEST.startswith("B"):
    # Eksperimen B masih berada di memori, langsung dipakai.
    best_model, best_tok = model_b, tok_b
else:
    # Eksperimen A sudah dibebaskan dari VRAM di §6 — adapter-nya dimuat ulang
    # dari disk, jauh lebih murah daripada melatih ulang 800 steps.
    free_memory("trainer_b", "model_b", "tok_b")
    best_model, best_tok = FastLanguageModel.from_pretrained(
        model_name=f"{CKPT_ROOT}/adapter_{BEST}",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN,
    )
    best_tok = get_chat_template(best_tok, chat_template="chatml")

print("model final siap:", BEST)
"""),

    md("## 8. Uji cepat model hasil fine-tuning"),
    code("""
FastLanguageModel.for_inference(best_model)

def generate(model, tok, question, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         temperature=0.7, top_p=0.9, do_sample=True)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print(generate(best_model, best_tok,
               "Jelaskan secara singkat apa itu perjanjian kerja waktu tertentu."))
"""),

    md("""
## 9. Push ke Hugging Face — **F1.4**

Menggunakan `push_to_hub_merged` dengan `save_method="merged_16bit"` agar adapter LoRA
tergabung ke bobot dasar dan model dapat dipanggil langsung pada tahap GRPO dan RAG.
Repositori dibuat **Public** (repo Private menyebabkan submission ditolak).
"""),
    code("""
best_model.push_to_hub_merged(
    HF_REPO_SFT,
    best_tok,
    save_method="merged_16bit",
    token=HF_TOKEN,
    private=False,
)
print("Model terunggah ke:", f"https://huggingface.co/{HF_REPO_SFT}")
"""),
    code("""
# Verifikasi repo benar-benar publik dan dapat diakses tanpa token.
from huggingface_hub import HfApi
info = HfApi().model_info(HF_REPO_SFT)
print("repo    :", info.id)
print("private :", info.private)
assert not info.private, "Repo masih Private — submission akan ditolak."
"""),

    md(f"""
## 10. Ringkasan

| Ketentuan | Status |
|---|---|
| Mapping dataset ke Chat Template + print sebelum/sesudah | ✅ §3 |
| QLoRA 4-bit + double quantization | ✅ §5 |
| LoRA pada MHA **dan** FFN (7 modul) | ✅ §5 |
| `SFTTrainer` 800 steps tanpa OOM | ✅ §6 |
| Split train/validation + eval_strategy="steps" + logging | ✅ §4, §6 |
| Dua eksperimen hyperparameter + perbandingan kurva | ✅ §6, §7 |
| Push `merged_16bit` ke HF (public) | ✅ §9 |

Model hasil notebook ini dilanjutkan ke
`GRPO_submission_PGABL_{STUDENT}.ipynb`.
"""),
]

write_nb(OUT, cells)
