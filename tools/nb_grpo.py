"""Bangun GRPO_submission_PGABL_<nama>.ipynb (Kriteria 1 level Advanced)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbutil import md, code, write_nb, STUDENT  # noqa: E402

OUT = pathlib.Path(__file__).parent.parent / f"GRPO_submission_PGABL_{STUDENT}.ipynb"

cells = [
    md(f"""
# GRPO — Melatih Model Bernalar `<think>`
**Proyek Akhir PGABL — {STUDENT.replace('-', ' ')}**

Notebook ini melanjutkan model instruct hasil `Fine-tuning_submission_PGABL_{STUDENT}.ipynb`
dan melatihnya dengan **GRPO** (`GRPOTrainer` dari TRL + Unsloth) agar model menuliskan
proses penalarannya di dalam tag `<think> ... </think>` sebelum memberi jawaban final.

| Kode | Ketentuan rubrik | Sel |
|------|------------------|-----|
| F1.7 | `GRPOTrainer` di atas model instruct hasil fine-tuning sendiri | §3, §6 |
| F1.8 | Empat reward function sesuai spesifikasi poin | §5 |
| F1.9 | `num_generations` & `max_completion_length` konservatif (mitigasi OOM) | §6 |
| F1.10 | Pengujian di pipeline RAG | notebook RAG |

Dataset tetap **`Ichsan2895/alpaca-gpt4-indonesian`** (dilarang memakai dataset lain).
"""),

    md("## 1. Instalasi dependensi"),
    code("""
!pip install -q -U "unsloth==2026.1.1" "trl==0.13.0" "peft==0.14.0" \\
    "transformers==4.48.2" "datasets==3.2.0" "accelerate==1.3.0" \\
    "bitsandbytes==0.45.0" "wandb==0.19.4" "rouge-score==0.1.2" "langdetect==1.0.9"

import torch, trl
print("torch:", torch.__version__, "| trl:", trl.__version__)
print("GPU  :", torch.cuda.get_device_name(0))
"""),

    md("## 2. Kredensial & konfigurasi"),
    code("""
import os

def load_secret(name, required=True):
    value = os.environ.get(name)
    try:
        from google.colab import userdata
        value = userdata.get(name) or value
    except Exception:
        pass
    if required and not value:
        raise RuntimeError(f"{name} belum di-set. Gunakan panel Secrets Colab.")
    if value:
        os.environ[name] = value
    print(f"{name}: {'tersedia' if value else 'tidak tersedia'}")
    return value

HF_TOKEN  = load_secret("HF_TOKEN")
WANDB_KEY = load_secret("WANDB_API_KEY", required=False)
"""),
    code("""
HF_USERNAME    = "vikaputri"                                  # <-- samakan dengan notebook 1
HF_REPO_SFT    = f"{HF_USERNAME}/qwen2.5-3b-legal-id-sft"     # input  (hasil fine-tuning)
HF_REPO_GRPO   = f"{HF_USERNAME}/qwen2.5-3b-legal-id-grpo"    # output (hasil GRPO)
DATASET_ID     = "Ichsan2895/alpaca-gpt4-indonesian"
MAX_SEQ_LENGTH = 1024
SEED           = 3407

USE_WANDB = bool(WANDB_KEY)
if USE_WANDB:
    import wandb
    wandb.login(key=WANDB_KEY, verify=True)
    os.environ["WANDB_PROJECT"] = "pgabl-legal-chatbot"
REPORT_TO = "wandb" if USE_WANDB else "none"
print("model dasar GRPO:", HF_REPO_SFT)
"""),

    md("""
## 3. Memuat kembali model instruct buatan sendiri — **F1.7**

Model yang dimuat adalah hasil fine-tuning pada notebook sebelumnya, bukan model
instruct pihak ketiga (larangan rubrik no. 7).
"""),
    code("""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HF_REPO_SFT,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
    fast_inference=False,
    token=HF_TOKEN,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)
print("model instruct siap untuk GRPO")
"""),

    md("""
## 4. Menyiapkan dataset prompt

GRPO hanya memerlukan **prompt**; jawabannya digenerate model sendiri lalu dinilai
reward function. Kolom `output` dataset tetap disimpan sebagai *ground truth* untuk
`correctness_reward`.
"""),
    code("""
from datasets import load_dataset

REASONING_SYSTEM_PROMPT = (
    "Anda adalah asisten AI Tim Legal berbahasa Indonesia.\\n"
    "Selalu jawab dengan format berikut:\\n"
    "<think>tuliskan proses penalaran Anda secara rinci di sini</think>\\n"
    "lalu tuliskan jawaban final Anda setelah tag penutup.\\n"
    "Gunakan Bahasa Indonesia sepenuhnya."
)

raw = load_dataset(DATASET_ID, split="train").shuffle(seed=SEED)
raw = raw.select(range(2000))     # cukup untuk ~250 steps GRPO, hemat waktu & VRAM

def to_grpo_format(example):
    user_content = example["instruction"]
    if example["input"]:
        user_content += f"\\n\\n{example['input']}"
    return {
        "prompt": [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "ground_truth": example["output"],
    }

grpo_dataset = raw.map(to_grpo_format, remove_columns=raw.column_names)
print(grpo_dataset)
print("\\nContoh prompt:")
print(grpo_dataset[0]["prompt"][1]["content"][:300])
"""),

    md("""
## 5. Reward Functions — **F1.8**

Empat fungsi berikut mengikuti spesifikasi poin pada rubrik secara persis.
"""),
    code("""
import re

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

def _texts(completions):
    # GRPOTrainer mengirim completions dalam format conversational.
    out = []
    for c in completions:
        if isinstance(c, list):
            out.append(c[0]["content"])
        elif isinstance(c, dict):
            out.append(c["content"])
        else:
            out.append(str(c))
    return out

def _reasoning_and_answer(text):
    # Mengembalikan (isi_think, jawaban_final).
    # Toleran bila </think> terpotong oleh max_completion_length.
    if THINK_OPEN not in text:
        return None, text.strip()
    after_open = text.split(THINK_OPEN, 1)[1]
    if THINK_CLOSE in after_open:
        reasoning, answer = after_open.split(THINK_CLOSE, 1)
        return reasoning, answer.strip()
    return after_open, ""      # reasoning terpotong, belum ada jawaban final
"""),
    md("""
### 5.1 `format_reward_func` — reward shaping bertahap (maks +1.0)

| Kondisi | Poin |
|---|---|
| Membuka respons dengan `<think>` | +0.2 |
| Menutup pemikiran dengan `</think>` | +0.3 |
| Format sempurna: tag di awal, tertutup benar, diikuti jawaban akhir | **+1.0** |
| Halusinasi: tag `<think>` atau `</think>` muncul lebih dari satu kali | **−0.5** |
"""),
    code("""
def format_reward_func(completions, **kwargs):
    rewards = []
    for text in _texts(completions):
        n_open  = text.count(THINK_OPEN)
        n_close = text.count(THINK_CLOSE)

        # Penalti halusinasi tag ganda mendahului penilaian lain.
        if n_open > 1 or n_close > 1:
            rewards.append(-0.5)
            continue

        stripped = text.lstrip()
        has_open  = n_open == 1
        has_close = n_close == 1
        answer_after_close = (
            stripped.split(THINK_CLOSE, 1)[1].strip() if has_close else ""
        )
        perfect = (
            stripped.startswith(THINK_OPEN)   # tag berada di awal kalimat
            and has_close                     # ditutup dengan benar
            and len(answer_after_close) > 0   # diikuti jawaban akhir
        )

        if perfect:
            rewards.append(1.0)
        else:
            score = 0.0
            if has_open:
                score += 0.2
            if has_close:
                score += 0.3
            rewards.append(score)
    return rewards
"""),
    md("""
### 5.2 `reasoning_length_reward` — memaksa penalaran benar-benar dijabarkan

| Panjang isi `<think>` | Poin |
|---|---|
| Tidak ada tag, atau isinya kosong / hanya spasi | +0.0 |
| < 50 karakter | +0.2 |
| 50 – 199 karakter | +0.5 |
| ≥ 200 karakter | **+1.0** |

Fungsi ini toleran bila `</think>` belum sempat muncul karena completion terpotong —
isi setelah `<think>` tetap dihitung.
"""),
    code("""
def reasoning_length_reward(completions, **kwargs):
    rewards = []
    for text in _texts(completions):
        reasoning, _ = _reasoning_and_answer(text)
        if reasoning is None or not reasoning.strip():
            rewards.append(0.0)
            continue
        n = len(reasoning.strip())
        if n < 50:
            rewards.append(0.2)
        elif n < 200:
            rewards.append(0.5)
        else:
            rewards.append(1.0)
    return rewards
"""),
    md("""
### 5.3 `correctness_reward` — kesesuaian dengan ground truth

Poin **+1.0** bila jawaban akhir memuat ground truth dari dataset, atau memiliki
kemiripan teks ROUGE-L di atas ambang.
"""),
    code("""
from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
ROUGE_THRESHOLD = 0.35

def _normalize(s):
    return re.sub(r"\\s+", " ", s.lower()).strip()

def correctness_reward(completions, ground_truth=None, **kwargs):
    texts = _texts(completions)
    truths = ground_truth if ground_truth is not None else [""] * len(texts)
    rewards = []
    for text, truth in zip(texts, truths):
        _, answer = _reasoning_and_answer(text)
        answer_n, truth_n = _normalize(answer), _normalize(truth or "")
        if not answer_n or not truth_n:
            rewards.append(0.0)
            continue
        if truth_n in answer_n:
            rewards.append(1.0)
            continue
        score = _scorer.score(truth_n, answer_n)["rougeL"].fmeasure
        rewards.append(1.0 if score >= ROUGE_THRESHOLD else 0.0)
    return rewards
"""),
    md("""
### 5.4 `language_reward_func` — menjaga jawaban tetap Bahasa Indonesia

Poin **−0.5** bila model beralih ke bahasa Inggris, **+1.0** bila jawaban akhir
murni Bahasa Indonesia.
"""),
    code("""
from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0

# Kata fungsi yang khas per bahasa, dipakai sebagai pemeriksa cepat sebelum langdetect.
ID_STOPWORDS = {"yang", "dan", "dengan", "untuk", "tidak", "adalah", "pada",
                "dari", "ini", "itu", "dapat", "akan", "atau", "karena"}
EN_STOPWORDS = {"the", "and", "with", "for", "not", "is", "are", "this",
                "that", "can", "will", "or", "because", "of"}

def _language_score(answer):
    words = re.findall(r"[a-zA-Z']+", answer.lower())
    if len(words) < 5:
        return 0.0                       # terlalu pendek untuk dinilai
    id_hits = sum(w in ID_STOPWORDS for w in words)
    en_hits = sum(w in EN_STOPWORDS for w in words)
    if en_hits > id_hits:
        return -0.5
    try:
        langs = {l.lang: l.prob for l in detect_langs(answer)}
    except Exception:
        return 1.0 if id_hits else 0.0
    if langs.get("en", 0.0) > 0.5:
        return -0.5
    if langs.get("id", 0.0) > 0.5:
        return 1.0
    return 0.0

def language_reward_func(completions, **kwargs):
    rewards = []
    for text in _texts(completions):
        _, answer = _reasoning_and_answer(text)
        rewards.append(_language_score(answer or text))
    return rewards
"""),

    md("### 5.5 Uji cepat keempat reward function"),
    code("""
CASES = {
    "format sempurna + penalaran panjang": (
        "<think>" + "Pekerja non-manajerial berhak atas upah lembur. " * 6 +
        "</think> Ya, Anda berhak menerima upah lembur sesuai peraturan yang berlaku."
    ),
    "tag ganda (halusinasi)": (
        "<think>alasan satu</think> jawaban <think>alasan dua</think> jawaban lagi"
    ),
    "think kosong": "<think>   </think> Jawaban tanpa penalaran apa pun di dalam tag.",
    "reasoning terpotong": "<think>Model sedang menjelaskan dasar hukumnya namun teksnya terpotong",
    "tanpa tag, bahasa Inggris": "The employee is entitled to overtime pay for the extra hours worked.",
}

import pandas as pd
rows = []
for label, text in CASES.items():
    comp = [[{"role": "assistant", "content": text}]]
    rows.append({
        "kasus": label,
        "format": format_reward_func(comp)[0],
        "reasoning_len": reasoning_length_reward(comp)[0],
        "correctness": correctness_reward(comp, ground_truth=["Ya, Anda berhak menerima upah lembur"])[0],
        "language": language_reward_func(comp)[0],
    })
display(pd.DataFrame(rows))
"""),

    md("""
## 6. Menjalankan GRPOTrainer — **F1.7, F1.9**

Mitigasi OOM: `num_generations=4` (kecil), `max_completion_length=256`,
`max_prompt_length=512`, dan batch per device 1 dengan gradient accumulation.
"""),
    code("""
from trl import GRPOConfig, GRPOTrainer

grpo_args = GRPOConfig(
    output_dir="outputs/grpo",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw_8bit",
    # per_device_train_batch_size WAJIB habis dibagi num_generations,
    # jadi 4 dipasangkan dengan num_generations=4 dan accumulation kecil.
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    # ---- F1.9: parameter penahan OOM ----
    num_generations=4,
    max_prompt_length=512,
    max_completion_length=256,
    max_steps=250,
    temperature=0.9,
    beta=0.04,
    logging_steps=5,
    save_steps=125,
    save_total_limit=1,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    seed=SEED,
    report_to=REPORT_TO,
    run_name="grpo-legal-id",
)

grpo_trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        reasoning_length_reward,
        correctness_reward,
        language_reward_func,
    ],
    args=grpo_args,
    train_dataset=grpo_dataset,
)
grpo_stats = grpo_trainer.train()
print(grpo_stats.metrics)
"""),

    md("## 7. Tren reward selama training"),
    code("""
import matplotlib.pyplot as plt

hist = [h for h in grpo_trainer.state.log_history if "reward" in h]
reward_cols = [k for k in hist[-1] if k.startswith("rewards/")] if hist else []

plt.figure(figsize=(11, 4.5))
plt.plot([h["step"] for h in hist], [h["reward"] for h in hist],
         label="total reward", linewidth=2)
for col in reward_cols:
    xs = [h["step"] for h in hist if col in h]
    ys = [h[col] for h in hist if col in h]
    plt.plot(xs, ys, alpha=0.65, label=col.replace("rewards/", ""))
plt.xlabel("step"); plt.ylabel("reward"); plt.grid(alpha=0.3)
plt.legend(); plt.title("Tren reward GRPO"); plt.tight_layout(); plt.show()
"""),

    md("## 8. Uji penalaran model hasil GRPO"),
    code("""
FastLanguageModel.for_inference(model)

def generate(question, max_new_tokens=384):
    messages = [
        {"role": "system", "content": REASONING_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         temperature=0.7, top_p=0.9, do_sample=True)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

jawaban = generate(
    "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. "
    "Apakah saya berhak dapat uang lembur?"
)
print(jawaban)
print("\\nMengandung tag <think>:", "<think>" in jawaban)
"""),

    md("""
## 9. Push model hasil GRPO ke Hugging Face

Sama seperti tahap SFT, memakai `merged_16bit` dan repositori **Public**.
"""),
    code("""
model.push_to_hub_merged(
    HF_REPO_GRPO,
    tokenizer,
    save_method="merged_16bit",
    token=HF_TOKEN,
    private=False,
)
print("Model GRPO terunggah ke:", f"https://huggingface.co/{HF_REPO_GRPO}")

from huggingface_hub import HfApi
info = HfApi().model_info(HF_REPO_GRPO)
print("private:", info.private)
assert not info.private, "Repo masih Private — submission akan ditolak."
"""),

    md(f"""
## 10. Ringkasan

| Ketentuan | Status |
|---|---|
| GRPOTrainer di atas model instruct buatan sendiri | ✅ §3, §6 |
| `format_reward_func` (+0.2 / +0.3 / +1.0, penalti −0.5) | ✅ §5.1 |
| `reasoning_length_reward` (0 / 0.2 / 0.5 / 1.0, toleran terpotong) | ✅ §5.2 |
| `correctness_reward` (ground truth atau ROUGE-L) | ✅ §5.3 |
| `language_reward_func` (−0.5 Inggris, +1.0 Indonesia) | ✅ §5.4 |
| `num_generations` & `max_completion_length` dikecilkan (anti-OOM) | ✅ §6 |
| Model menampilkan `<think>` sebelum jawaban final | ✅ §8 |

Model hasil notebook ini dipakai sebagai generator pada
`RAG_submission_PGABL_{STUDENT}.ipynb` (pemenuhan **F1.10**).
"""),
]

write_nb(OUT, cells)
