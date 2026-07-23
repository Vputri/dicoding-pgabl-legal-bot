# PRD — Fine-tuned Chatbot Tim Legal berbasis RAG

**Proyek Akhir — Dicoding: Pengembangan Generative AI berbasis LLM (PGABL)**
Penulis: Vika Putri Ariyanti
Tanggal: 20 Juli 2026
Target nilai: **Bintang 5 / A / Advanced (4 pts)**

---

## 1. Latar Belakang

Tim Legal perusahaan menangani volume dokumen hukum yang besar — Undang-Undang, Peraturan Pemerintah, dan dokumen kepatuhan internal — sebagian besar berupa PDF panjang dan berstruktur kompleks. Pencarian manual memakan waktu, sementara LLM publik (ChatGPT, Gemini) tidak dapat dipakai karena dua alasan:

1. **Kerahasiaan** — dokumen hukum internal tidak boleh keluar dari lingkungan perusahaan.
2. **Akurasi** — model publik tidak mengenal regulasi spesifik yang dipakai perusahaan, sehingga rawan menghasilkan jawaban yang meyakinkan tetapi salah (halusinasi).

## 2. Tujuan Produk

Membangun asisten AI internal yang menjawab pertanyaan hukum **cepat, akurat, dan dapat dipertanggungjawabkan** — berbasis dokumen resmi milik perusahaan, bukan spekulasi model.

### Sasaran terukur

| # | Sasaran | Indikator keberhasilan |
|---|---------|------------------------|
| G1 | Model punya kemampuan tanya-jawab instruksional berbahasa Indonesia | SFT selesai ≥ 800 steps tanpa OOM; kurva loss turun & stabil |
| G2 | Jawaban bersumber dokumen resmi | Setiap jawaban menyertakan sitasi (nama UU + halaman) |
| G3 | Model menunjukkan penalaran sebelum menjawab | Output memuat blok `<think>…</think>` yang terisi substantif |
| G4 | Sistem tahu kapan dokumen lokal tidak memadai | Skor reranker Top-1 < threshold → fallback ke web search |
| G5 | Dapat dipakai non-teknis | Antarmuka Gradio satu input–satu output |

### Non-Goals

- Tidak membangun API produksi, autentikasi, atau deployment berskala.
- Tidak memberi nasihat hukum final — output adalah bantuan riset, bukan pengganti advokat.
- Tidak memakai model atau embedding proprietary (OpenAI, Cohere, dll.) di titik mana pun.

## 3. Pengguna & Skenario

**Persona utama:** anggota Tim Legal (non-engineer) yang perlu jawaban cepat berbasis regulasi.

Skenario inti (test case wajib submission):

> **Prompt:** "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?"
>
> **Output diharapkan:**
> `<think>` User adalah staf admin (pekerja non-manajerial). Berdasarkan PP 35/2021, pekerja yang bekerja melebihi waktu kerja standar wajib dibayar upah lembur. 3 jam lembur harus dibayar. `</think>` Ya, Anda berhak. Berdasarkan PP No. 35 Tahun 2021, perusahaan wajib membayar upah lembur untuk staf admin yang bekerja melebihi waktu kerja normal (8 jam sehari).
>
> *Sumber: PP No. 35 Tahun 2021, hal. X*

## 4. Batasan Wajib (Constraints)

Batasan ini bersifat *hard requirement* dari rubrik — pelanggaran mana pun menyebabkan **Reject (0 pts)**.

| Aspek | Ketentuan |
|-------|-----------|
| Bahasa | Python |
| Dataset SFT & GRPO | `Ichsan2895/alpaca-gpt4-indonesian` — format Alpaca, bahasa Indonesia |
| Dokumen RAG | 4 file PDF UU dari Google Drive yang disediakan — **seluruhnya wajib dipakai** |
| Base model | Task *Text Generation*, arsitektur didukung Unsloth (Llama / Mistral / Qwen / Gemma / Phi), dari penyedia terpercaya atau kuantisasi resmi Unsloth |
| Fine-tuning | QLoRA 4-bit + **double quantization**; LoRA adapter minimal pada satu komponen komputasi utama penuh (MHA atau FFN) |
| Training | `SFTTrainer` minimal **800 steps**, selesai tanpa OOM |
| Upload | `model.push_to_hub(...)` metode **`merged_16bit`** ke Hugging Face |
| Embedding | Wajib open-source (mis. `intfloat/multilingual-e5`, `BAAI/bge-m3`) |
| Vector DB | Lokal — ChromaDB atau FAISS |
| Generator RAG | Wajib model hasil Kriteria 1 — bukan model baru dari HF, bukan proprietary |
| Chunking | Ukuran chunk & overlap dinyatakan **eksplisit**; chunk ≤ 5000 |
| Notebook | Semua sel sudah dijalankan, output tersimpan, bebas error |

### 4.1 Larangan — penyebab Submission Ditolak

Sembilan kondisi berikut membuat submission **ditolak reviewer**, terlepas dari kualitas teknis lainnya:

1. Tidak melampirkan file yang diminta pada ketentuan berkas (§8).
2. Terdeteksi memakai platform **No-Code/Low-Code atau AutoML** (mis. Hugging Face AutoTrain) atau UI tools instan untuk fine-tuning. → Seluruh training wajib ditulis sebagai kode Python di notebook.
3. Notebook **tidak dijalankan** lebih dulu sehingga output sel tidak terekam.
4. Tidak mengimplementasikan fitur wajib sesuai level secara berurutan (Basic → Skilled → Advanced).
5. Memakai dokumen lain, atau hanya **sebagian** dari 4 PDF yang disediakan, sebagai knowledge RAG.
6. Memakai dataset selain `Ichsan2895/alpaca-gpt4-indonesian` untuk fine-tuning maupun GRPO.
7. Memakai model instruct/chat hasil fine-tuning **pihak ketiga atau siswa lain** di Hugging Face.
8. Tautan model HF bersifat **Private**, salah ketik, atau tidak disertakan. → Repo wajib **Public** dan tautannya diverifikasi dari jendela incognito sebelum submit.
9. **Tidak menyembunyikan API key** (Hugging Face Token, WandB) — wajib lewat environment variable.

### 4.2 Pengelolaan Secret

Token tidak boleh muncul sebagai literal di sel notebook maupun di output sel.

```python
# Colab: simpan di panel Secrets (ikon kunci), bukan hard-code
from google.colab import userdata
import os

os.environ["HF_TOKEN"]    = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

Sebelum submit, telusuri notebook untuk memastikan tidak ada string `hf_…` atau key WandB yang tersisa di source maupun output — termasuk output `!env`, `wandb.login()`, atau traceback.

### 4.3 Logging Eksperimen (WandB)

Perbandingan dua eksperimen hyperparameter (F1.6) dicatat via Weights & Biases:

- Daftar/masuk di wandb.ai → menu **API Keys** → buat token.
- Simpan token sebagai `WANDB_API_KEY` di Colab Secrets (§4.2), lalu set `report_to="wandb"` dan `run_name` berbeda per eksperimen.
- Alternatif bila tidak memakai WandB: `report_to="none"` dan plot kurva loss manual dari `trainer.state.log_history` — yang dinilai adalah kurva dan analisisnya, bukan tool-nya.

## 5. Ruang Lingkup Fungsional

### 5.1 Kriteria 1 — Fine-tuning SLM

**Basic (2 pts)**
- **F1.1** Mapping dataset mentah → Chat Template Unsloth (Llama-3 / ChatML) via `datasets.map`. Wajib `print` satu baris hasil mapping lengkap dengan token spesial, dan tampilkan pula contoh baris sebelum mapping sebagai pembanding.
- **F1.2** Muat model 4-bit QLoRA dengan `bnb_4bit_use_double_quant=True`. Definisikan LoRA pada minimal satu komponen penuh:
  - MHA: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - FFN: `gate_proj`, `up_proj`, `down_proj`
  Direkomendasikan keduanya (7 modul) untuk kualitas terbaik.
- **F1.3** Jalankan `SFTTrainer` ≥ 800 steps sampai selesai, tanpa OOM.
- **F1.4** `push_to_hub_merged(..., save_method="merged_16bit")` ke repo HF publik.

**Skilled (3 pts)** — semua Basic terpenuhi, ditambah:
- **F1.5** Split dataset `train` / `validation`; `TrainingArguments` memuat `eval_dataset`, `eval_strategy="steps"`, `eval_steps`, `logging_steps`.
- **F1.6** Minimal **2 eksperimen training** dengan kombinasi hyperparameter berbeda (mis. `learning_rate`, `lora_r`/`lora_alpha`, `batch_size`). Sertakan tabel perbandingan train vs eval loss dan plot kurva, lalu argumentasikan kombinasi mana yang terbaik tanpa overfitting.

**Advanced (4 pts)** — semua Skilled terpenuhi, ditambah:
- **F1.7** GRPO dengan `GRPOTrainer` (TRL + Unsloth) di atas model instruct hasil SFT (muat ulang dari HF).
- **F1.8** Empat reward function:

  | Fungsi | Aturan poin |
  |--------|-------------|
  | `format_reward_func` | Buka `<think>` → +0.2 · Tutup `</think>` → +0.3 · Format sempurna (tag di awal, tertutup benar, diikuti jawaban akhir) → +1.0 · Penalti **−0.5** bila tag `<think>`/`</think>` muncul lebih dari sekali |
  | `reasoning_length_reward` | Tanpa tag / isi kosong-spasi → +0.0 · `<` 50 karakter → +0.2 · 50–199 karakter → +0.5 · ≥ 200 karakter → +1.0. Harus toleran jika reasoning terpotong `max_completion_length` |
  | `correctness_reward` | +1.0 bila jawaban akhir memuat ground truth kolom `output`, atau kemiripan ROUGE/BLEU melewati ambang |
  | `language_reward_func` | **−0.5** bila jawaban beralih ke bahasa Inggris · +1.0 bila murni Bahasa Indonesia |

- **F1.9** Atur `num_generations` dan `max_completion_length` konservatif untuk memitigasi OOM.
- **F1.10** Uji model hasil GRPO di dalam pipeline RAG memakai test case wajib (§3); output harus menampilkan `<think>` sebelum jawaban final berbasis dokumen retrieved.

### 5.2 Kriteria 2 — Sistem RAG

**Basic (2 pts)**
- **F2.1** Muat 4 PDF; split dengan text splitter, `chunk_size` & `chunk_overlap` eksplisit (usulan: 1000 / 150).
- **F2.2** Embedding open-source → simpan ke FAISS/ChromaDB lokal.
- **F2.3** Muat model hasil Kriteria 1; susun prompt berisi `{context}` dan `{question}`; jalankan generation.
- **F2.4** Bungkus dalam antarmuka sederhana — `gr.Interface` satu input–satu output (dipilih), atau loop `input()` + `IPython.display.Markdown`.

**Skilled (3 pts)** — semua Basic terpenuhi, ditambah:
- **F2.5** *Metadata enrichment* per chunk: nama UU, nomor & tahun, nama file, nomor halaman, nomor pasal/bab bila terdeteksi.
- **F2.6** *Metadata filtering* (mis. batasi ke UU tertentu) + **sitasi** pada setiap jawaban.
- **F2.7** *Ensemble Retriever*: BM25 (keyword) + FAISS/Chroma (semantik) dengan bobot eksplisit (usulan 0.4 / 0.6), `k ≥ 5`.
- **F2.8** *Parent-Child Retriever*: child chunk kecil untuk pencarian vektor, parent chunk besar (halaman utuh) sebagai konteks LLM.

**Advanced (4 pts)** — semua Skilled terpenuhi, ditambah:
- **F2.9** **HyDE** — LLM membuat **minimal 2** jawaban hipotetis atas query, di-embed, dipakai sebagai query transformation.
- **F2.10** **Reranker** cross-encoder (mis. `BAAI/bge-reranker-v2-m3`) → ambil Top-K = 3 chunk paling relevan.
- **F2.11** **Adaptive fallback** — ekstrak relevance score reranker pada dokumen Top-1; jika `< threshold`, abaikan dokumen lokal dan panggil DuckDuckGo Search untuk mencari dari internet. Threshold ditetapkan eksplisit dan alasannya dijelaskan.

## 6. Arsitektur Sistem

```
                       ┌──────────────────────────────────────┐
  4 PDF UU  ──────────▶│ Load → Enrich metadata → Parent/Child │
                       │        splitting (chunk+overlap)      │
                       └───────────────┬──────────────────────┘
                                       ▼
                        Embedding open-source → FAISS/Chroma (lokal)
                                       │
  Query user ──▶ HyDE (≥2 jawaban hipotetis) ──▶ Ensemble Retriever
                                       │          (BM25 + semantik, bobot)
                                       ▼
                             Reranker cross-encoder → Top-3
                                       │
                        skor Top-1 ≥ threshold ?
                          ┌────────────┴────────────┐
                        ya│                         │tidak
                          ▼                         ▼
                Parent chunks sbg konteks     DuckDuckGo Search
                          └────────────┬────────────┘
                                       ▼
              Prompt {context}+{question} ──▶ Model SFT+GRPO (Kriteria 1)
                                       ▼
                     <think> reasoning </think> jawaban + sitasi
                                       ▼
                              Gradio gr.Interface
```

## 7. Rencana Pengerjaan

| Tahap | Aktivitas | Output |
|-------|-----------|--------|
| T1 | Setup Colab/Kaggle GPU, pin versi Unsloth/TRL/transformers | environment stabil |
| T2 | Load + mapping dataset Alpaca ke chat template, print before/after | F1.1 |
| T3 | Eksperimen SFT #1 (baseline) | kurva loss #1 |
| T4 | Eksperimen SFT #2 (hyperparameter berbeda) + perbandingan | F1.5, F1.6 |
| T5 | Push model terbaik `merged_16bit` ke HF | F1.4 |
| T6 | GRPO + 4 reward function di atas model SFT, push hasil | F1.7–F1.9 |
| T7 | Ingest 4 PDF, metadata enrichment, parent-child, index vektor | F2.1, F2.2, F2.5, F2.8 |
| T8 | Ensemble retriever + HyDE + reranker + fallback DuckDuckGo | F2.7, F2.9–F2.11 |
| T9 | Prompt template, sitasi, Gradio, jalankan test case wajib | F2.3, F2.4, F2.6, F1.10 |
| T10 | Rerun notebook end-to-end, simpan output, susun berkas & zip | deliverable |

## 8. Deliverable

Seluruh pekerjaan dikumpulkan dalam **satu folder ter-zip**:

```
PGABL_Vika-Putri-Ariyanti.zip
├── Fine-tuning_submission_PGABL_Vika-Putri-Ariyanti.ipynb
├── GRPO_submission_PGABL_Vika-Putri-Ariyanti.ipynb
├── RAG_submission_PGABL_Vika-Putri-Ariyanti.ipynb
├── link_huggingface.txt
└── requirements.txt
```

Catatan berkas:
- Notebook GRPO opsional secara rubrik, **tetapi wajib** untuk target Advanced.
- Semua `.ipynb` harus **sudah dijalankan** dan menyimpan output — reviewer tidak perlu menjalankan ulang.
- `link_huggingface.txt` memuat tautan repo model hasil fine-tuning (dan hasil GRPO).
- `requirements.txt` memuat versi terkunci seluruh dependensi.

### 8.1 Menyiapkan `requirements.txt`

| Cara | Perilaku | Catatan |
|------|----------|---------|
| `pip freeze > requirements.txt` | Semua library terpasang di environment beserta versinya | Lengkap tapi berisik — di Colab ikut menyertakan ratusan paket bawaan |
| `pipreqs /path/to/project` | Hanya library yang benar-benar di-`import` di kode | Ringkas, tapi bisa melewatkan dependensi tak langsung |

Rekomendasi: mulai dari `pipreqs`, lalu pin manual versi kritikal (`unsloth`, `trl`, `transformers`, `peft`, `bitsandbytes`, `langchain`, `faiss-cpu`/`chromadb`, `sentence-transformers`, `rank_bm25`, `gradio`, `duckduckgo-search`) dari hasil `pip freeze` — supaya reviewer bisa mereproduksi tanpa konflik versi.

### 8.2 Ekspor Notebook

Di Google Colab: **File → Download → Download .ipynb**. Pastikan diekspor **setelah** semua sel dijalankan agar output ikut tersimpan. Ganti nama file sesuai konvensi §8, kumpulkan dalam satu folder, lalu zip.

### 8.3 Proses Review

- Review memakan waktu selambatnya **3 hari kerja** (di luar Sabtu, Minggu, dan libur nasional).
- Hindari submit berulang kali — memperlambat proses penilaian.
- Notifikasi hasil dikirim via email atau dapat dicek pada status submission di akun Dicoding.
- Kesulitan teknis ditanyakan ke forum diskusi kelas.

## 9. Definition of Done

- [ ] Ketiga notebook jalan penuh tanpa error, output tersimpan
- [ ] Output mapping dataset (sebelum & sesudah) tercetak dengan token spesial
- [ ] QLoRA 4-bit + double quantization aktif, LoRA menutupi ≥ 1 komponen penuh
- [ ] `SFTTrainer` selesai ≥ 800 steps, tanpa OOM
- [ ] Dua eksperimen hyperparameter terdokumentasi + plot kurva loss & analisis overfitting
- [ ] Model ter-push ke HF via `merged_16bit`
- [ ] GRPO berjalan dengan 4 reward function sesuai spesifikasi poin
- [ ] Keempat PDF terindeks; chunk size & overlap eksplisit
- [ ] Embedding & vector DB open-source/lokal
- [ ] Ensemble retriever berbobot (`k ≥ 5`), parent-child retriever, metadata filter, sitasi
- [ ] HyDE ≥ 2 jawaban hipotetis; reranker Top-3; fallback DuckDuckGo berbasis threshold
- [ ] Gradio berjalan, test case lembur menghasilkan `<think>` + jawaban bersitasi
- [ ] Zip lengkap sesuai struktur berkas §8
- [ ] Repo HF **Public** dan tautannya diverifikasi dari incognito
- [ ] Tidak ada API key/token terlihat di source maupun output sel
- [ ] Fine-tuning murni kode Python — tanpa AutoTrain/AutoML/UI tools
- [ ] Keempat PDF resmi terpakai, tanpa dokumen tambahan di luar yang disediakan
- [ ] `requirements.txt` memuat versi terkunci dependensi kritikal

## 10. Risiko & Mitigasi

| Risiko | Dampak | Mitigasi |
|--------|--------|----------|
| OOM saat SFT | Kriteria 1 Reject | 4-bit + `gradient_checkpointing="unsloth"`, `per_device_train_batch_size` kecil + gradient accumulation, `max_seq_length` secukupnya |
| OOM saat GRPO | Kehilangan Advanced | Turunkan `num_generations` (mis. 4) & `max_completion_length` (mis. 256); pakai model SFT yang lebih kecil bila perlu |
| Sesi Colab terputus | Kehilangan progres | Checkpoint berkala ke Drive, `resume_from_checkpoint`, push adapter lebih awal |
| Model beralih ke bahasa Inggris | Melanggar tujuan produk | `language_reward_func` di GRPO + instruksi sistem berbahasa Indonesia pada prompt RAG |
| Reranker menandai semua dokumen relevan padahal tidak | Halusinasi | Kalibrasi threshold dari sampel query in/out-of-scope, dokumentasikan angkanya |
| Ketidakcocokan versi Unsloth/TRL | Notebook error saat direview | Pin versi di `requirements.txt` dan rerun penuh sebelum zip |
| PDF hasil scan / teks berantakan | Retrieval buruk | Verifikasi ekstraksi teks per file; ganti loader (PyMuPDF/pdfplumber) bila perlu |
| Token bocor di output sel | **Ditolak** | Secret via `userdata.get()`; audit source & output sebelum ekspor (§4.2) |
| Repo HF ter-set Private | **Ditolak** | Set Public saat push; verifikasi tautan dari incognito |
| Memuat ulang model berkali-kali dalam satu notebook | VRAM terbuang → OOM | Restart runtime setiap ganti model bahasa; hindari re-load di sel yang sama |

## 11. Metrik Evaluasi Kualitas

- **Training:** train vs eval loss per step; gap yang melebar = sinyal overfitting.
- **GRPO:** tren rata-rata tiap reward function sepanjang training.
- **RAG:** retrieval hit-rate pada ~10 pertanyaan uji buatan sendiri; keakuratan sitasi (UU & halaman benar); persentase jawaban yang memuat `<think>` terisi.

---

*Nilai akhir = Total Points / Jumlah Kriteria. Target: 4 pts (Kriteria 1) + 4 pts (Kriteria 2) ÷ 2 = **4,0 → Bintang 5 / A / Advanced**.*
