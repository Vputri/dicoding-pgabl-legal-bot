# Proyek Akhir PGABL — Fine-tuned Chatbot Tim Legal berbasis RAG

Implementasi dari [PRD-Legal-Chatbot-RAG.md](../PRD-Legal-Chatbot-RAG.md).
Target: **Advanced (4 pts)** pada kedua kriteria.

## Isi

| Berkas | Kriteria | Isi |
|---|---|---|
| `Fine-tuning_submission_PGABL_Vika-Putri-Ariyanti.ipynb` | K1 Basic–Skilled | Mapping ChatML, QLoRA 4-bit + double quant, SFT 800 steps, 2 eksperimen hyperparameter, push `merged_16bit` |
| `GRPO_submission_PGABL_Vika-Putri-Ariyanti.ipynb` | K1 Advanced | GRPOTrainer + 4 reward function, mitigasi OOM, push `merged_16bit` |
| `RAG_submission_PGABL_Vika-Putri-Ariyanti.ipynb` | K2 Basic–Advanced | Ingest 4 PDF, metadata enrichment, parent-child, ensemble BM25+FAISS, HyDE, reranker, fallback DuckDuckGo, Gradio |
| `requirements.txt` | — | Dependensi terkunci versinya |
| `link_huggingface.txt` | — | Tautan repo model (**wajib diisi & public**) |
| `tools/` | — | Skrip pembangun & validator notebook — **tidak disertakan ke dalam zip** |

## Urutan menjalankan

Ketiganya dijalankan di **Google Colab dengan GPU** (T4 cukup), berurutan karena saling bergantung:

1. **Fine-tuning** → menghasilkan repo `…-sft` di Hugging Face.
2. **GRPO** → memuat repo `…-sft`, menghasilkan repo `…-grpo`.
3. **RAG** → memuat repo `…-grpo` sebagai generator.

### Sebelum menjalankan

1. **Set username Hugging Face.** Ganti `HF_USERNAME = "vikaputri"` di §2 ketiga notebook agar sama.
2. **Simpan token di Colab Secrets** (ikon kunci di panel kiri), bukan di dalam sel:
   - `HF_TOKEN` — token Hugging Face dengan izin *write*
   - `WANDB_API_KEY` — opsional; bila kosong, logging otomatis jatuh ke plot manual
3. **Siapkan 4 PDF** dari tautan Drive "Dokumen Knowledge RAG" ke folder Drive
   `MyDrive/Dokumen Knowledge RAG/`. Notebook RAG menyalinnya ke `/content/dokumen_uu`
   dan **menggagalkan eksekusi bila jumlahnya bukan 4**.
4. Notebook fine-tuning menjalankan dua kali training 800 steps — sediakan waktu
   yang cukup, dan aktifkan checkpoint bila sesi Colab rawan terputus.

### Setelah menjalankan

1. Isi `link_huggingface.txt` dengan URL repo yang sebenarnya.
2. Verifikasi kedua repo **Public** dari jendela incognito.
3. Unduh notebook dari Colab (**File → Download → Download .ipynb**) **setelah**
   semua sel dijalankan agar outputnya ikut tersimpan.
4. Jalankan pemeriksaan akhir dan buat zip:

```bash
python3 tools/validate_notebooks.py   # cek JSON, sintaks, dan kebocoran token
python3 tools/test_rewards.py         # cek reward function vs spesifikasi rubrik
python3 tools/package.py              # buat PGABL_Vika-Putri-Ariyanti.zip
```

## Keputusan desain

- **Base model `unsloth/Qwen2.5-3B-bnb-4bit`** — varian *base* (bukan instruct) dari repo
  kuantisasi resmi Unsloth. Dipilih agar kemampuan mengikuti instruksi benar-benar
  berasal dari fine-tuning sendiri, sekaligus menghindari tuduhan memakai model
  instruct pihak ketiga (larangan rubrik no. 7).
- **LoRA pada 7 modul** (MHA *dan* FFN) — rubrik hanya mewajibkan satu komponen penuh;
  keduanya dipakai untuk kualitas yang lebih baik.
- **Chunk 400/80 (child) dan 2000/200 (parent)** — child kecil membuat embedding lebih
  presisi, parent besar menjaga konteks pasal tetap utuh saat masuk ke LLM.
- **Bobot ensemble 0.4 BM25 : 0.6 FAISS** — BM25 diperlukan untuk istilah yang harus
  persis ("Pasal 78"), FAISS untuk parafrase bahasa sehari-hari.
- **Threshold reranker 0.30** (skor sigmoid) — angka awal; **kalibrasi ulang** memakai
  pertanyaan in-scope vs out-of-scope Anda sendiri sebelum submit, lalu perbarui
  penjelasannya di §8 notebook RAG.

## Yang masih perlu dikerjakan sendiri

Notebook di folder ini **belum dijalankan** — outputnya masih kosong, dan submission
dengan output kosong akan ditolak. Yang wajib dilakukan:

- [ ] Jalankan ketiga notebook sampai selesai di Colab GPU
- [ ] Push kedua model, set repo ke Public
- [ ] Isi `link_huggingface.txt`
- [ ] Kalibrasi `RELEVANCE_THRESHOLD` dari hasil nyata
- [ ] Tulis ulang analisis di §7 notebook fine-tuning sesuai kurva loss yang benar-benar diperoleh
- [ ] Unduh .ipynb ber-output, lalu buat zip
