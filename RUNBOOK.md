# Runbook — menjalankan ketiga notebook di Google Colab

Tujuan: mengubah notebook yang outputnya masih kosong menjadi notebook ber-output
yang siap dizip. **Wajib lewat Colab/Kaggle** — Unsloth butuh GPU NVIDIA (CUDA),
tidak berjalan di Mac Apple Silicon.

Perkiraan total waktu: **4–7 jam**, sebagian besar menunggu training.

---

## Persiapan (sekali saja, ~20 menit)

### 1. Akun & token Hugging Face

1. Daftar di https://huggingface.co
2. **Settings → Access Tokens → Create new token**
3. Tipe **Write** (bukan Read — notebook perlu mengunggah model)
4. Salin tokennya, dan **catat username HF Anda**

### 2. Siapkan dokumen di Google Drive

Unduh 4 PDF dari tautan "Dokumen Knowledge RAG", lalu taruh di Drive pada:

```
MyDrive/Dokumen Knowledge RAG/*.pdf
```

Notebook RAG membaca folder ini. Kalau nama foldernya berbeda, ubah perintah `cp` di §2.

### 3. Samakan username di notebook

Ganti `HF_USERNAME = "vikaputri"` menjadi username HF Anda di **§2 ketiga notebook**.
Harus sama persis di tiganya — notebook GRPO dan RAG memuat model dari repo yang
dibuat notebook sebelumnya.

### 4. Unggah notebook ke Colab

https://colab.research.google.com → **File → Upload notebook** → pilih ketiga `.ipynb`.

### 5. Set Secrets di Colab

Ikon **kunci** di panel kiri → tambahkan, dan aktifkan *Notebook access*:

| Nama | Isi | Wajib |
|---|---|---|
| `HF_TOKEN` | token write dari langkah 1 | ya |
| `WANDB_API_KEY` | token wandb.ai | tidak — bila kosong, notebook otomatis pakai plot manual |

Secrets berlaku per-akun, jadi cukup diisi sekali untuk ketiga notebook.

### 6. Aktifkan GPU

**Runtime → Change runtime type → T4 GPU → Save.** Lakukan di setiap notebook.

---

## Menjalankan (berurutan — tidak boleh dibalik)

### Notebook 1 — Fine-tuning (~3–5 jam)

**Runtime → Run all.** Yang perlu diperhatikan:

- Sel Drive akan meminta izin akses — setujui. Checkpoint disimpan ke
  `MyDrive/pgabl_outputs` agar selamat bila sesi terputus.
- Dua eksperimen × 800 steps. Ini bagian terlama.
- Sel terakhir memverifikasi repo HF tidak private.

**Kalau terputus di tengah:** jalankan ulang dari atas. Sel training mendeteksi
checkpoint dan melanjutkan, tidak mengulang dari nol.

**Kalau kehabisan kuota GPU:** ganti `BASE_MODEL` ke `unsloth/Qwen2.5-1.5B-bnb-4bit`
di §2 (kira-kira 2× lebih cepat, tetap memenuhi rubrik), lalu ulangi.

**Kalau OOM:** turunkan `MAX_SEQ_LENGTH` ke 512, atau
`per_device_train_batch_size` ke 1 sambil menaikkan `gradient_accumulation_steps`.

Hasil: repo `…-sft` di Hugging Face.

### Notebook 2 — GRPO (~1–2 jam)

Pastikan notebook 1 sudah selesai dan repo `…-sft` ada. **Runtime → Run all.**

- §5.5 mencetak tabel uji reward function — pastikan angkanya masuk akal.
- §8 harus menampilkan jawaban dengan tag `<think>`. Kalau belum muncul, naikkan
  `max_steps` GRPO dari 250.

**Kalau OOM:** turunkan `num_generations` ke 2 dan `per_device_train_batch_size`
ke 2 (keduanya harus tetap habis dibagi).

Hasil: repo `…-grpo` di Hugging Face.

### Notebook 3 — RAG (~30–45 menit)

Pastikan repo `…-grpo` ada. **Runtime → Run all.**

- Sel §3 **sengaja gagal** bila PDF yang terbaca bukan tepat 4 — periksa folder Drive.
- §8 memakai `RELEVANCE_THRESHOLD = 0.30`. **Kalibrasi ulang**: lihat skor Top-1 yang
  tercetak di §11(a) (pertanyaan in-scope, seharusnya tinggi) dan §11(b) (out-of-scope,
  seharusnya rendah), lalu set ambang di antaranya dan perbarui penjelasan di §8.
- §12 adalah test case wajib — output harus memuat `<think>` dan sitasi.
- §13 memunculkan tautan Gradio. Biarkan output tautannya tersimpan sebagai bukti.

---

## Setelah semua selesai

1. **Unduh ketiganya:** File → Download → Download .ipynb — **setelah** semua sel jalan.
2. **Timpa** file di folder ini dengan hasil unduhan (nama file jangan diubah).
3. **Isi `link_huggingface.txt`** — ganti `<username>`, hapus blok `=== BELUM DIISI ===`.
4. **Verifikasi kedua repo HF public** dari jendela incognito.
5. Jalankan pemeriksaan dan pengemasan:

```bash
python3 tools/validate_notebooks.py   # JSON, sintaks, kebocoran token
python3 tools/test_rewards.py         # reward function vs rubrik
python3 tools/package.py              # baru akan jadi bila semua syarat lolos
```

`package.py` menolak membuat zip selama masih ada sel tanpa output atau placeholder
di `link_huggingface.txt` — itu memang disengaja, supaya Anda tidak submit dalam
kondisi yang pasti ditolak.

---

## Sebelum submit — periksa manual

- [ ] Tidak ada token yang terlihat di output sel mana pun (cek output `wandb.login`,
      traceback, dan sel Drive)
- [ ] Analisis di §7 notebook 1 sudah ditulis ulang sesuai kurva loss yang **benar-benar**
      Anda peroleh — teks bawaan hanya kerangka
- [ ] Penjelasan threshold di §8 notebook 3 sesuai angka hasil kalibrasi Anda
- [ ] Zip berisi tepat 5 berkas, tanpa folder `tools/`
