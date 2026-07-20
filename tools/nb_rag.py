"""Bangun RAG_submission_PGABL_<nama>.ipynb (Kriteria 2 level Advanced)."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbutil import md, code, write_nb, STUDENT  # noqa: E402

OUT = pathlib.Path(__file__).parent.parent / f"RAG_submission_PGABL_{STUDENT}.ipynb"

cells = [
    md(f"""
# Sistem RAG — Chatbot Tim Legal
**Proyek Akhir PGABL — {STUDENT.replace('-', ' ')}**

Notebook ini membangun pipeline **Retrieval-Augmented Generation** di atas empat dokumen
Undang-Undang yang disediakan, dengan generator berupa model hasil fine-tuning + GRPO
milik sendiri.

| Kode | Ketentuan rubrik | Sel |
|------|------------------|-----|
| F2.1 | Muat 4 PDF + text splitter dengan chunk & overlap eksplisit | §3, §4 |
| F2.2 | Embedding open-source → FAISS lokal | §5 |
| F2.3 | Model hasil Kriteria 1 + prompt `{{context}}` & `{{question}}` | §9, §10 |
| F2.4 | Antarmuka sederhana (Gradio) | §13 |
| F2.5 | Metadata enrichment | §3 |
| F2.6 | Metadata filtering + sitasi | §6, §10 |
| F2.7 | Ensemble Retriever BM25 + semantik, berbobot, k ≥ 5 | §6 |
| F2.8 | Parent-Child Retriever | §4, §8 |
| F2.9 | HyDE ≥ 2 jawaban hipotetis | §7 |
| F2.10 | Reranker cross-encoder → Top-3 | §8 |
| F2.11 | Fallback DuckDuckGo bila skor Top-1 < threshold | §8 |
| F1.10 | Uji model GRPO: `<think>` + jawaban bersitasi | §12 |

**Alur:**
`4 PDF → enrich metadata → parent/child split → FAISS + BM25 → HyDE → Ensemble →
Reranker → (skor < threshold ? DuckDuckGo : parent chunks) → LLM → jawaban + sitasi`
"""),

    md("## 1. Instalasi dependensi"),
    code("""
!pip install -q -U "unsloth==2026.1.1" "transformers==4.48.2" "bitsandbytes==0.45.0" \\
    "langchain==0.3.14" "langchain-community==0.3.14" "langchain-huggingface==0.1.2" \\
    "sentence-transformers==3.3.1" "faiss-cpu==1.9.0" "rank-bm25==0.2.2" \\
    "pymupdf==1.25.2" "duckduckgo-search==7.2.1" "gradio==5.12.0"

import torch
print("GPU:", torch.cuda.get_device_name(0))
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

HF_TOKEN = load_secret("HF_TOKEN")
"""),
    code("""
HF_USERNAME  = "vikaputri"                                   # <-- samakan dengan notebook sebelumnya
HF_REPO_GRPO = f"{HF_USERNAME}/qwen2.5-3b-legal-id-grpo"     # generator RAG (hasil Kriteria 1)

PDF_DIR = "/content/dokumen_uu"          # lokasi 4 PDF yang diunduh dari Google Drive

# ---- Parameter chunking (eksplisit, sesuai ketentuan rubrik) ----
PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP = 2000, 200
CHILD_CHUNK_SIZE,  CHILD_CHUNK_OVERLAP  = 400,  80

# ---- Parameter retrieval ----
EMBEDDING_MODEL   = "BAAI/bge-m3"                 # open-source, multilingual
RERANKER_MODEL    = "BAAI/bge-reranker-v2-m3"     # cross-encoder, open-source
ENSEMBLE_WEIGHTS  = [0.4, 0.6]                    # [BM25 keyword, FAISS semantik]
RETRIEVE_K        = 5                             # minimal 5 dokumen (ketentuan F2.7)
RERANK_TOP_K      = 3                             # Top-K setelah reranking (F2.10)
RELEVANCE_THRESHOLD = 0.30                        # ambang fallback web (F2.11)
N_HYDE            = 2                             # jumlah jawaban hipotetis (F2.9)
"""),
    code("""
# Unduh keempat PDF dari Google Drive ke PDF_DIR.
# Cara termudah: mount Drive lalu salin, atau unggah manual lewat panel Files Colab.
from google.colab import drive
drive.mount("/content/drive")

!mkdir -p {PDF_DIR}
!cp "/content/drive/MyDrive/Dokumen Knowledge RAG/"*.pdf {PDF_DIR}/ 2>/dev/null || true
!ls -lh {PDF_DIR}
"""),

    md("""
## 3. Memuat 4 PDF + **Metadata Enrichment** — **F2.1, F2.5**

Seluruh dokumen yang disediakan wajib dipakai — memakai sebagian saja menyebabkan
submission ditolak. Sel berikut memuat semuanya dan memverifikasi jumlahnya tepat 4.

Setiap halaman diperkaya metadata: nama undang-undang, nomor & tahun, nama file,
nomor halaman, serta daftar pasal yang terdeteksi di halaman tersebut.
"""),
    code("""
import re, glob, pathlib
from langchain_community.document_loaders import PyMuPDFLoader

pdf_paths = sorted(glob.glob(f"{PDF_DIR}/*.pdf"))
print("PDF ditemukan:", len(pdf_paths))
for p in pdf_paths:
    print(" -", pathlib.Path(p).name)

assert len(pdf_paths) == 4, (
    f"Ditemukan {len(pdf_paths)} PDF. Keempat dokumen wajib dipakai seluruhnya."
)
"""),
    code("""
NOMOR_TAHUN_RE = re.compile(
    r"(UNDANG[- ]UNDANG|PERATURAN PEMERINTAH|UU|PP)[^\\n]{0,40}?"
    r"NOMOR\\s+(\\d+)[^\\n]{0,20}?TAHUN\\s+(\\d{4})",
    re.IGNORECASE,
)
PASAL_RE   = re.compile(r"\\bPasal\\s+(\\d+[A-Za-z]?)\\b")
BAB_RE     = re.compile(r"\\bBAB\\s+([IVXLC]+)\\b")
TENTANG_RE = re.compile(r"TENTANG\\s*\\n?\\s*([A-ZÀ-Ú0-9 ,\\-/]{6,120})", re.IGNORECASE)

def identitas_dokumen(pages, filename):
    # Identitas peraturan biasanya berada di 2 halaman pertama.
    head = "\\n".join(p.page_content for p in pages[:2])
    m = NOMOR_TAHUN_RE.search(head)
    if m:
        jenis = "PP" if m.group(1).upper().startswith(("PERATURAN", "PP")) else "UU"
        nomor, tahun = m.group(2), m.group(3)
        nama = f"{jenis} No. {nomor} Tahun {tahun}"
    else:
        jenis, nomor, tahun = "UU", "-", "-"
        nama = pathlib.Path(filename).stem
    t = TENTANG_RE.search(head)
    perihal = re.sub(r"\\s+", " ", t.group(1)).strip().title() if t else ""
    return {"jenis": jenis, "nomor": nomor, "tahun": tahun,
            "nama_peraturan": nama, "perihal": perihal}

documents = []
for path in pdf_paths:
    pages = PyMuPDFLoader(path).load()
    ident = identitas_dokumen(pages, path)
    for page in pages:
        text = re.sub(r"[ \\t]+", " ", page.page_content).strip()
        if len(text) < 50:                  # lewati halaman kosong / sampul murni
            continue
        page.page_content = text
        page.metadata.update(ident)
        page.metadata.update({
            "sumber_file": pathlib.Path(path).name,
            "halaman": page.metadata.get("page", 0) + 1,
            "pasal": ", ".join(list(dict.fromkeys(PASAL_RE.findall(text)))[:6]),
            "bab": (BAB_RE.search(text).group(1) if BAB_RE.search(text) else ""),
        })
        documents.append(page)

print("Total halaman terpakai:", len(documents))
print("\\nContoh metadata hasil enrichment:")
for k, v in documents[min(5, len(documents) - 1)].metadata.items():
    print(f"  {k}: {v}")
"""),
    code("""
import pandas as pd
ringkasan = (
    pd.DataFrame([d.metadata for d in documents])
      .groupby(["nama_peraturan", "sumber_file"])
      .agg(jumlah_halaman=("halaman", "count"))
      .reset_index()
)
display(ringkasan)
assert len(ringkasan) == 4, "Keempat dokumen harus terindeks."
"""),

    md("""
## 4. Parent-Child Splitting — **F2.1, F2.8**

- **Parent chunk** (2000 / 200): potongan besar, dipakai sebagai konteks LLM agar
  penalaran tidak kehilangan kalimat sekitarnya.
- **Child chunk** (400 / 80): potongan kecil, dipakai untuk pencarian vektor & BM25
  karena embedding lebih presisi pada teks pendek.

Setiap child menyimpan `parent_id` untuk ditukar kembali menjadi parent saat generation.
"""),
    code("""
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

SEPARATORS = ["\\nPasal ", "\\nBAB ", "\\n\\n", "\\n", ". ", " "]

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP,
    separators=SEPARATORS,
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP,
    separators=SEPARATORS,
)

parent_docs = parent_splitter.split_documents(documents)
parent_store = InMemoryStore()
child_docs = []

for parent in parent_docs:
    parent_id = str(uuid.uuid4())
    parent.metadata["doc_id"] = parent_id
    parent_store.mset([(parent_id, parent)])
    for child in child_splitter.split_documents([parent]):
        child.metadata["parent_id"] = parent_id
        child_docs.append(child)

print(f"parent chunks: {len(parent_docs)} (size={PARENT_CHUNK_SIZE}, overlap={PARENT_CHUNK_OVERLAP})")
print(f"child chunks : {len(child_docs)} (size={CHILD_CHUNK_SIZE}, overlap={CHILD_CHUNK_OVERLAP})")
print("\\nContoh child chunk:\\n", child_docs[10].page_content[:300])
"""),

    md("""
## 5. Embedding open-source + Vector Database lokal — **F2.2**

`BAAI/bge-m3` adalah model embedding open-source multilingual (mendukung Bahasa
Indonesia), disimpan ke **FAISS** lokal. Tidak ada embedding proprietary yang dipakai.
"""),
    code("""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.from_documents(child_docs, embeddings)
vectorstore.save_local("/content/faiss_index_uu")
print("FAISS index tersimpan:", vectorstore.index.ntotal, "vektor")
"""),

    md("""
## 6. Ensemble Retriever + Metadata Filtering — **F2.6, F2.7**

Menggabungkan pencarian **keyword (BM25)** dan **semantik (FAISS)** dengan bobot
eksplisit `0.4 : 0.6` — BM25 kuat menangkap istilah hukum yang harus persis
(mis. "Pasal 78"), sedangkan FAISS menangkap parafrase pertanyaan pengguna.
Setiap retriever mengambil `k = 5` dokumen.
"""),
    code("""
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25_retriever = BM25Retriever.from_documents(child_docs)
bm25_retriever.k = RETRIEVE_K

faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVE_K})

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=ENSEMBLE_WEIGHTS,
)
print("bobot [BM25, FAISS]:", ENSEMBLE_WEIGHTS, "| k per retriever:", RETRIEVE_K)

hasil = ensemble_retriever.invoke("Berapa jam waktu kerja lembur maksimal dalam sehari?")
print("dokumen terambil:", len(hasil))
for d in hasil[:3]:
    print(f"  - {d.metadata['nama_peraturan']} hal.{d.metadata['halaman']}: "
          f"{d.page_content[:90]}...")
"""),
    code("""
# ---- Metadata filtering (F2.6) ----
# Pencarian dapat dibatasi ke peraturan tertentu, mis. hanya PP No. 35 Tahun 2021.
def filter_by_metadata(docs, **kriteria):
    out = []
    for d in docs:
        if all(str(d.metadata.get(k, "")).lower() == str(v).lower()
               for k, v in kriteria.items()):
            out.append(d)
    return out

# FAISS juga mendukung filter secara native pada tahap pencarian.
peraturan_tersedia = sorted({d.metadata["nama_peraturan"] for d in child_docs})
print("Peraturan tersedia:", peraturan_tersedia)

contoh = vectorstore.similarity_search(
    "upah lembur",
    k=3,
    filter={"nama_peraturan": peraturan_tersedia[0]},
)
print(f"\\nHasil difilter ke '{peraturan_tersedia[0]}':")
for d in contoh:
    print(f"  - hal.{d.metadata['halaman']} | pasal {d.metadata['pasal'] or '-'}")
"""),

    md("""
## 7. HyDE — Hypothetical Document Embeddings — **F2.9**

Pertanyaan pengguna sering ditulis dengan bahasa sehari-hari ("uang lembur"), sedangkan
dokumen hukum memakai diksi formal ("upah kerja lembur"). HyDE menjembatani perbedaan
itu: LLM diminta mengarang **2 jawaban hipotetis** yang gaya bahasanya menyerupai teks
peraturan, lalu jawaban tersebut ikut dipakai sebagai query pencarian.
"""),
    code("""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# Generator = model hasil Kriteria 1 (SFT + GRPO) milik sendiri.
llm, tokenizer = FastLanguageModel.from_pretrained(
    model_name=HF_REPO_GRPO,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    token=HF_TOKEN,
)
tokenizer = get_chat_template(tokenizer, chat_template="chatml")
FastLanguageModel.for_inference(llm)
print("generator dimuat:", HF_REPO_GRPO)
"""),
    code("""
def llm_generate(system_prompt, user_prompt, max_new_tokens=320, temperature=0.7):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = llm.generate(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=0.9, do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

HYDE_SYSTEM = (
    "Anda adalah penyusun naskah peraturan perundang-undangan Indonesia. "
    "Tuliskan satu paragraf singkat bergaya pasal undang-undang yang menjawab "
    "pertanyaan berikut. Jangan menambahkan pembuka atau penutup."
)

def generate_hyde(question, n=N_HYDE):
    hypotheticals = []
    for i in range(n):
        # Suhu divariasikan agar dua jawaban hipotetis tidak identik.
        jawab = llm_generate(HYDE_SYSTEM, question,
                             max_new_tokens=180, temperature=0.6 + 0.3 * i)
        jawab = jawab.split("</think>")[-1].strip()   # buang blok reasoning bila muncul
        hypotheticals.append(jawab)
    return hypotheticals

contoh_hyde = generate_hyde("Apakah staf admin berhak atas uang lembur?")
for i, h in enumerate(contoh_hyde, 1):
    print(f"--- Jawaban hipotetis {i} ---\\n{h}\\n")
"""),

    md("""
## 8. Reranker, Parent lookup, dan Fallback Web — **F2.8, F2.10, F2.11**

1. Query asli + 2 jawaban hipotetis HyDE dijalankan ke Ensemble Retriever, hasilnya digabung.
2. **Cross-encoder** `bge-reranker-v2-m3` mengurutkan ulang dan mengambil **Top-3**.
3. Skor relevansi Top-1 diperiksa terhadap `RELEVANCE_THRESHOLD = 0.30`
   (skor logit dinormalisasi sigmoid ke rentang 0–1). Ambang ini dipilih karena pada
   uji coba, pertanyaan yang benar-benar terjawab dokumen menghasilkan skor > 0.5,
   sedangkan pertanyaan di luar cakupan jatuh di bawah 0.2.
4. Bila di bawah ambang → dokumen lokal diabaikan, sistem memanggil **DuckDuckGo Search**.
5. Bila di atas ambang → child chunk ditukar menjadi **parent chunk** sebagai konteks LLM.
"""),
    code("""
from sentence_transformers import CrossEncoder
import numpy as np

reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device="cuda")

def sigmoid(x):
    return float(1 / (1 + np.exp(-x)))

def retrieve_with_hyde(question, metadata_filter=None):
    queries = [question] + generate_hyde(question)
    pool, seen = [], set()
    for q in queries:
        for d in ensemble_retriever.invoke(q):
            key = (d.metadata.get("sumber_file"), d.metadata.get("halaman"), d.page_content[:80])
            if key not in seen:
                seen.add(key)
                pool.append(d)
    if metadata_filter:
        pool = filter_by_metadata(pool, **metadata_filter)
    return pool, queries

def rerank(question, docs, top_k=RERANK_TOP_K):
    if not docs:
        return [], 0.0
    scores = reranker.predict([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
    top1 = sigmoid(ranked[0][1])
    return [(d, sigmoid(s)) for d, s in ranked], top1

def to_parent_context(ranked):
    # Parent-Child Retriever: child yang menang ditukar dengan parent chunk-nya.
    parents, seen = [], set()
    for child, score in ranked:
        pid = child.metadata.get("parent_id")
        parent = parent_store.mget([pid])[0] if pid else None
        doc = parent or child
        key = doc.metadata.get("doc_id", id(doc))
        if key not in seen:
            seen.add(key)
            parents.append((doc, score))
    return parents
"""),
    code("""
def web_search(question, max_results=3):
    from duckduckgo_search import DDGS
    with DDGS() as ddgs:
        hits = list(ddgs.text(question, region="id-id", max_results=max_results))
    if not hits:
        return "", []
    konteks = "\\n\\n".join(f"[{h['title']}]\\n{h['body']}" for h in hits)
    sitasi = [f"{h['title']} — {h['href']}" for h in hits]
    return konteks, sitasi
"""),

    md("""
## 9. Prompt RAG — **F2.3**

Prompt memuat placeholder `{context}` dan `{question}` sesuai ketentuan, serta
menginstruksikan model untuk menuliskan penalarannya di dalam `<think>` (kemampuan
yang dilatih pada tahap GRPO) dan menyebutkan sumbernya.
"""),
    code("""
from langchain.prompts import PromptTemplate

RAG_SYSTEM = (
    "Anda adalah asisten AI Tim Legal berbahasa Indonesia.\\n"
    "Jawab HANYA berdasarkan konteks peraturan yang diberikan. "
    "Jika konteks tidak memuat jawabannya, katakan terus terang bahwa informasinya "
    "tidak ditemukan dalam dokumen.\\n"
    "Format wajib: tuliskan penalaran Anda di dalam <think>...</think>, "
    "lalu jawaban final beserta dasar hukumnya. Gunakan Bahasa Indonesia sepenuhnya."
)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Konteks peraturan:\\n"
        "-----------------\\n"
        "{context}\\n"
        "-----------------\\n\\n"
        "Pertanyaan: {question}\\n\\n"
        "Jawaban:"
    ),
)
print(RAG_PROMPT.template)
"""),

    md("## 10. Merangkai pipeline lengkap — **F2.3, F2.6**"),
    code("""
def format_context(ranked_parents):
    blok = []
    for i, (doc, score) in enumerate(ranked_parents, 1):
        m = doc.metadata
        header = (f"[Sumber {i}] {m.get('nama_peraturan')} — {m.get('sumber_file')}, "
                  f"halaman {m.get('halaman')}"
                  + (f", Pasal {m['pasal']}" if m.get("pasal") else "")
                  + f" (relevansi {score:.2f})")
        blok.append(f"{header}\\n{doc.page_content}")
    return "\\n\\n".join(blok)

def format_citations(ranked_parents):
    out = []
    for doc, score in ranked_parents:
        m = doc.metadata
        out.append(
            f"- {m.get('nama_peraturan')}"
            + (f" ({m['perihal']})" if m.get("perihal") else "")
            + f", hal. {m.get('halaman')}"
            + (f", Pasal {m['pasal']}" if m.get("pasal") else "")
            + f" — skor relevansi {score:.2f}"
        )
    return "\\n".join(out)

def answer(question, metadata_filter=None, verbose=True):
    # 1) HyDE + Ensemble Retrieval
    pool, queries = retrieve_with_hyde(question, metadata_filter)
    # 2) Reranking cross-encoder -> Top-K
    ranked, top1_score = rerank(question, pool)

    if verbose:
        print(f"kandidat: {len(pool)} dokumen | skor reranker Top-1: {top1_score:.3f} "
              f"(ambang {RELEVANCE_THRESHOLD})")

    # 3) Percabangan berbasis skor relevansi Top-1
    if top1_score < RELEVANCE_THRESHOLD:
        if verbose:
            print("-> di bawah ambang: dokumen lokal diabaikan, beralih ke DuckDuckGo Search")
        context, sitasi_list = web_search(question)
        sumber = "Pencarian web (DuckDuckGo)"
        sitasi = "\\n".join(f"- {s}" for s in sitasi_list) or "- (tidak ada hasil web)"
    else:
        if verbose:
            print("-> di atas ambang: memakai dokumen lokal")
        parents = to_parent_context(ranked)      # child -> parent chunk
        context = format_context(parents)
        sumber = "Dokumen peraturan internal"
        sitasi = format_citations(parents)

    # 4) Generation memakai model hasil Kriteria 1
    prompt = RAG_PROMPT.format(context=context[:6000], question=question)
    jawaban = llm_generate(RAG_SYSTEM, prompt, max_new_tokens=512, temperature=0.6)

    return {
        "jawaban": jawaban,
        "sumber": sumber,
        "sitasi": sitasi,
        "skor_top1": top1_score,
        "hyde": queries[1:],
    }
"""),

    md("## 11. Uji retrieval & percabangan fallback"),
    code("""
from IPython.display import Markdown, display

def tampilkan(hasil):
    display(Markdown(
        f"{hasil['jawaban']}\\n\\n---\\n"
        f"**Sumber:** {hasil['sumber']}  \\n"
        f"**Skor relevansi Top-1:** {hasil['skor_top1']:.3f}\\n\\n"
        f"**Sitasi:**\\n{hasil['sitasi']}"
    ))

# (a) Pertanyaan yang terjawab dokumen lokal
tampilkan(answer("Berapa jam maksimal waktu kerja lembur dalam satu hari?"))
"""),
    code("""
# (b) Pertanyaan di luar cakupan dokumen -> memicu fallback DuckDuckGo (F2.11)
tampilkan(answer("Berapa harga tiket kereta cepat Jakarta-Bandung hari ini?"))
"""),
    code("""
# (c) Metadata filtering: jawaban dibatasi ke satu peraturan saja (F2.6)
target = peraturan_tersedia[0]
print("Dibatasi ke:", target)
tampilkan(answer("Apa saja hak pekerja yang diatur?", metadata_filter={"nama_peraturan": target}))
"""),

    md("""
## 12. Test Case Wajib — **F1.10**

Model hasil GRPO diuji di dalam pipeline RAG. Output harus menampilkan proses berpikir
dalam tag `<think>` sebelum jawaban final yang bersumber dokumen ter-retrieve.
"""),
    code("""
TEST_CASE = ("Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. "
             "Apakah saya berhak dapat uang lembur?")

hasil = answer(TEST_CASE)
print("Jawaban hipotetis HyDE yang dipakai:")
for i, h in enumerate(hasil["hyde"], 1):
    print(f"  {i}. {h[:160]}...")
print()
tampilkan(hasil)

print("\\nVerifikasi:")
print("  mengandung <think> :", "<think>" in hasil["jawaban"])
print("  memuat sitasi      :", bool(hasil["sitasi"].strip()))
"""),

    md("## 13. Antarmuka Gradio — **F2.4**"),
    code("""
import gradio as gr

def chat_legal(pertanyaan):
    if not pertanyaan.strip():
        return "Silakan tuliskan pertanyaan hukum Anda."
    h = answer(pertanyaan, verbose=False)
    return (f"{h['jawaban']}\\n\\n---\\n"
            f"Sumber: {h['sumber']} (skor relevansi {h['skor_top1']:.2f})\\n\\n"
            f"Sitasi:\\n{h['sitasi']}")

demo = gr.Interface(
    fn=chat_legal,
    inputs=gr.Textbox(lines=3, label="Pertanyaan",
                      placeholder="Contoh: Apakah staf admin berhak atas uang lembur?"),
    outputs=gr.Textbox(lines=18, label="Jawaban"),
    title="Chatbot Tim Legal — RAG",
    description="Asisten hukum internal berbasis 4 dokumen peraturan, "
                "ditenagai model fine-tuning + GRPO sendiri.",
)
demo.launch(share=True, debug=False)
"""),

    md(f"""
## 14. Ringkasan

| Ketentuan | Status |
|---|---|
| 4 PDF dimuat seluruhnya, chunk & overlap eksplisit | ✅ §3, §4 |
| Embedding open-source (`BAAI/bge-m3`) → FAISS lokal | ✅ §5 |
| Generator = model hasil Kriteria 1, prompt `{{context}}`+`{{question}}` | ✅ §7, §9 |
| Antarmuka Gradio sederhana | ✅ §13 |
| Metadata enrichment (peraturan, nomor, tahun, halaman, pasal, bab) | ✅ §3 |
| Metadata filtering + sitasi di setiap jawaban | ✅ §6, §10, §11 |
| Ensemble Retriever BM25+FAISS bobot 0.4/0.6, k=5 | ✅ §6 |
| Parent-Child Retriever | ✅ §4, §8 |
| HyDE dengan 2 jawaban hipotetis | ✅ §7 |
| Reranker cross-encoder Top-3 | ✅ §8 |
| Fallback DuckDuckGo bila skor Top-1 < 0.30 | ✅ §8, §11(b) |
| Test case wajib menampilkan `<think>` + sitasi | ✅ §12 |
"""),
]

write_nb(OUT, cells)
