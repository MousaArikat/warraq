# Warraq (ورّاق) — Arabic Document Intelligence
### DESIGN.md — v0.1 (Source of Truth)

> Codename "Warraq" (the medieval Arabic scribe/manuscript copyist). Final name TBD.
> ⚠️ Naming note: "Baseer" is taken (Misraj AI's commercial product, Dec 2025). Avoid it.
> Candidate alternatives: Warraq (ورّاق), Raqeem (رقيم — "inscription"), Bayan (بيان).

---

## 1. Mission

An **open, benchmarked, grounded** Arabic document understanding system:

1. **Read**: Convert scanned/photographed Arabic documents (print, forms, tables, light
   handwriting) into clean structured Markdown/JSON — powered by a **fine-tuned small
   open VLM (~3–4B)** that beats or matches frontier APIs on Arabic documents.
2. **Ground**: Answer questions about a document with **pixel-level visual citations** —
   the exact region on the scan is highlighted (bounding boxes), not just a text quote.
3. **Abstain**: Calibrated refusal. When the document is illegible or the answer isn't
   present, the system says so. Selective prediction is a first-class feature.
4. **Serve**: Quantized, served with vLLM, deployed via Helm/GitOps on Kubernetes,
   instrumented with OpenTelemetry, with an eval-gated CI pipeline.

**Headline goal (the wow line for the README):**
> "A fine-tuned 3B open model that beats GPT-4o and Gemini on scanned Arabic
> documents — with public benchmark numbers, visual citations, and honest abstention.
> Total training cost: under $500."

**Commercial wedge (if this becomes a startup):** Data sovereignty. MENA governments,
courts, banks, and law firms cannot ship documents to US APIs. A small model with
near-frontier Arabic document accuracy that runs **on-prem / in-region on Kubernetes**
is worth more to them than GPT-4o. Warraq is the open proof-of-capability for exactly
that pitch.

---

## 2. Non-Goals (scope guardrails — re-read when tempted)

- ❌ NOT a general Arabic chatbot. Documents only.
- ❌ NOT full historical-manuscript / heavy-calligraphy OCR (stretch goal only).
- ❌ NOT all dialects / speech / audio. Text documents only.
- ❌ NOT a multi-tenant SaaS with auth/billing. Single-user demo product.
- ❌ NOT training from scratch. We fine-tune an existing open VLM.
- ❌ No real personal documents EVER in data or demos. Synthetic identities only.

---

## 3. Success Criteria

**Technical**
- [ ] Eval harness reproduces KITAB-Bench protocol on ≥3 task types (OCR/page→MD, tables, extraction).
- [ ] Fine-tuned model beats its own base model by a large, documented margin on all tasks.
- [ ] Fine-tuned model **beats at least one frontier API overall**, and beats ALL frontier
      APIs on at least one defined slice (e.g., degraded scans, Arabic forms). Honest
      reporting either way — losses are published too.
- [ ] Grounded QA: answers carry bbox citations; citation hit-rate ≥85% on eval set.
- [ ] Abstention: risk–coverage curve published; hallucinated answers on unanswerable
      questions <10% at the chosen operating point.
- [ ] Post-quantization accuracy delta measured and published.
- [ ] Full system runs on k3d via `helm install`, with OTel traces end-to-end.

**Career / portfolio**
- [ ] README leaderboard table with reproducible numbers + demo GIF.
- [ ] Model + synthetic dataset released on Hugging Face.
- [ ] Technical report (8–12 pages, arXiv-style PDF or long-form blog post).
- [ ] One interview-ready sentence per phase ("I built a synthetic data engine that...",
      "I fine-tuned and quantized...", "I built an eval-gated CI...").

---

## 4. Prior Art & Positioning (know the landscape — be honest about it)

| Who | What | Our relationship to it |
|---|---|---|
| **KITAB-Bench** (MBZUAI, ACL 2025) | The Arabic OCR/doc benchmark. 8,809 samples, 9 domains. Best model (Gemini 2.0 Flash) ≈65% on PDF→Markdown. | Our primary eval. We report on its subsets. |
| **Cross-Lingual SynthDocs** (Nov 2025, arXiv 2511.04699) | Synthetic Arabic doc corpus → fine-tuned Qwen2.5-VL-3B → big gains across benchmarks. | Our proven training recipe. Replicate, then extend. |
| **Qari-OCR** (NAMAA, HF) | Qwen2-VL fine-tunes for Arabic OCR. | Prior art + baseline to compare against. Proof the approach works. |
| **AIN** (MBZUAI) | Arabic-centric multimodal LMM. | Baseline / possible alternative base model. |
| **Misraj Baseer, Sanad.ai, Doxci** | Closed commercial Arabic IDP products. | Market validation. They're closed; we're open + grounded + benchmarked. |
| **CAMEL-Bench** (MBZUAI) | Arabic LMM benchmark; shows handwriting/historical is where everything fails. | Secondary eval; informs our "hard slice". |

**Differentiation = the combination:** open weights + public benchmark numbers +
pixel-level grounded citations + calibrated abstention + production-grade serving.
No existing open project has all five.

---

## 5. System Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │                  FRONTEND                    │
                        │   Next.js (RTL-first) · scan viewer with     │
                        │   bbox highlight overlays · confidence UI    │
                        └──────────────────────┬──────────────────────┘
                                               │ REST
                        ┌──────────────────────┴──────────────────────┐
                        │              API (FastAPI)                   │
                        │  /ingest  /documents  /query  /health        │
                        └──────┬───────────────────────────┬──────────┘
                               │                           │
                ┌──────────────┴─────────────┐   ┌─────────┴──────────────┐
                │      INGESTION PIPELINE     │   │      QUERY PIPELINE     │
                │ 1. PDF/image → page rasters │   │ 1. Embed query (BGE-M3) │
                │ 2. (opt) layout detect      │   │ 2. Retrieve chunks      │
                │ 3. Warraq-VLM OCR via vLLM  │   │ 3. Rerank               │
                │    → markdown + word bboxes │   │ 4. Generate answer w/   │
                │ 4. Chunk (bbox metadata)    │   │    citations → bboxes   │
                │ 5. Embed + index            │   │ 5. Confidence gate →    │
                │                             │   │    answer OR abstain    │
                └──────────────┬─────────────┘   └─────────┬──────────────┘
                               │                           │
                ┌──────────────┴───────────────────────────┴──────────┐
                │                    CORE SERVICES                     │
                │  vLLM (Warraq-VLM, quantized) · embedding server ·   │
                │  vector store (qdrant or pgvector) · object storage  │
                └──────────────────────────┬──────────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                │                 INFRA (Phase 4)                      │
                │  k3d/kind · Helm charts · FluxCD GitOps · OTel →     │
                │  Prometheus/Grafana · GitHub Actions eval-gate CI    │
                └─────────────────────────────────────────────────────┘

   OFFLINE (Phases 0–2):
   data-engine (synth render + degrade + QA gen) ──► training (QLoRA) ──► quantize
                          ▲                                │
                          └────────── eval harness ◄───────┘   (every checkpoint)
```

---

## 6. PHASE 0 — Evaluation Harness (2–3 weeks) ★ DO THIS FIRST

**Goal:** A measuring stick + baseline leaderboard before any training.

### 6.1 Eval datasets
- **KITAB-Bench subsets** (HF: datasets are `ahmedheakl/arocrbench_*` — a 24-item
  collection under user `ahmedheakl`, NOT `mbzuai-oryx` which only hosts the eval *code*
  at github.com/mbzuai-oryx/KITAB-Bench. Use the GitHub repo's prompts + metric protocol
  so our numbers are paper-comparable): page OCR (image→text), PDF→Markdown,
  tables, optionally charts. Use their published splits; do NOT train on eval splits.
- **Warraq-Eval (ours, ~150–200 pages), built in Phase 1's generator + manual curation:**
  - 40 clean printed MSA pages (books/articles, multiple fonts)
  - 40 degraded scans (same pages through degradation pipeline + a few real public scans)
  - 40 synthetic forms (key-value ground truth JSON, fake identities)
  - 30 tables (financial/government style, HTML ground truth)
  - 20 mixed AR/EN business docs (invoices/contracts style)
  - 10–20 handwriting (from public datasets: KHATT / Muharaf) — labeled "hard slice"
- **QA eval set:** 5 questions per doc on a 50-doc subset: 3 answerable (with gold answer
  + gold bbox region), 2 unanswerable (for abstention measurement).

### 6.2 Metrics (implement in `evals/metrics.py`)
- **CER / WER** via `jiwer`, with an **Arabic normalization layer** (critical!):
  - report both *with* and *without* diacritics (strip tashkeel variant)
  - normalize alef variants (أ إ آ ا), ta marbuta/ha, alef maqsura/ya — as configurable flags
  - normalize Arabic-Indic digits (٠١٢٣) ↔ Western digits — configurable
  - strip tatweel (ـ)
- **Tables:** TEDS (tree edit distance similarity) on HTML.
- **Key-value extraction:** field-level exact match + ANLS.
- **Markdown structure:** heading/list/table presence score (simple structural F1; mirror
  KITAB-Bench's protocol where feasible).
- **QA:** exact match + token F1; **citation hit-rate** (predicted bbox IoU≥0.5 with gold
  region); **abstention:** precision/recall on unanswerable set + risk–coverage curve.

### 6.3 Baselines (runners in `evals/baselines/`)
- APIs: GPT-4o (or current frontier), Gemini Flash (current), Claude (current).
- Open: Qwen2.5-VL-3B & 7B (zero-shot), Qari-OCR checkpoint, AIN if runnable.
- Traditional: Tesseract (ara), Surya, EasyOCR, PaddleOCR — to show the gap.

### 6.4 Deliverables
- `warraq evals run --model <name> --suite <suite>` CLI.
- `evals/report.py` → regenerates the README leaderboard markdown from results JSON.
- Results committed under `evals/results/` (JSON, one file per run, with git SHA + config).
- W&B (free tier) or MLflow project for tracking. Decision: **W&B** (nicer reports).

### 6.5 Definition of Done
Leaderboard table in README with ≥6 baselines across ≥3 suites. Blog-able on its own
("How good are frontier models at Arabic documents? I measured.").

### 6.6 Rigor Rules — public-claim insurance (non-negotiable)
Every published number must survive hostile review. These rules are CI-enforced or
checklist-enforced before anything goes in the README:

1. **No contamination.** Never train on benchmark eval splits. Warraq-Eval source
   texts/pages hash-excluded from the synthetic generator. Exclusion mechanism documented.
2. **No cherry-picking.** All suites reported, including the ones we lose. Wins and
   losses in the same table.
3. **Fair prompting.** Identical prompt per suite across all models (or documented
   best-effort per model). Exact prompts committed to the repo.
4. **Real sample sizes.** ≥200 samples per headline suite; N reported in every table.
5. **Reproducible by strangers.** One command re-runs any number; pinned dependency
   versions; configs + results JSON + git SHA committed.
6. **Variance handled.** Final published API numbers = mean of 3 runs (temperature 0
   where supported); variance noted.
7. **No metric gaming.** Normalization flags documented; CER reported both with and
   without diacritic stripping.
8. **Test the tests.** Unit tests for every metric against hand-computed golden values;
   bbox pipelines verified with visual overlay checks.
9. **Auditability.** Raw model outputs cached/stored so metrics can be recomputed
   without re-calling APIs.
10. **Human spot-check.** ≥30 random samples manually reviewed per major run to confirm
    metrics match perceived quality.

---

## 7. PHASE 1 — Synthetic Data Engine (3–4 weeks) ★ THE MOAT

**Goal:** 50K–200K (image, ground-truth) pairs with FREE perfect labels including
word-level bounding boxes. Quality > quantity; start at 20K, scale what helps.

### 7.1 Text corpora (all open)
- Arabic Wikipedia dumps; **Hindawi** books (public domain); **OpenITI** corpus
  (classical); **Tashkeela** (diacritized text — gives us diacritics-rich pages);
  Arabic news datasets (e.g., SANAD); Jordanian/Arab legal texts (official gazettes,
  public laws); UN Arabic documents. Discovery resource: **Masader** catalog (Arabic
  NLP datasets index).
- Fake entities: Arabic name lists + Faker-style generators for IDs, dates, amounts,
  addresses. **Never real people.**

### 7.2 Rendering pipeline (`data/synth/render.py`)
- HTML/CSS templates → render via **Playwright** (headless Chromium; best Arabic
  shaping/bidi support) or WeasyPrint. Output PNG @ ~150–300 DPI equivalent.
- Template variety: book page, two-column article, government form, invoice, contract,
  memo, table-heavy report, certificate. 15–25 templates.
- **Fonts (≥20):** Amiri, Noto Naskh Arabic, Noto Kufi, Cairo, Tajawal, Scheherazade New,
  Lateef, IBM Plex Sans Arabic, Markazi, Almarai, Aref Ruqaa (display), Reem Kufi, etc.
- Variations: font size/weight, line height, mixed AR/EN spans, Arabic-Indic vs Western
  digits, with/without tashkeel (sample from Tashkeela), headers/footers/page numbers,
  stamps/signature placeholders (images), watermarks.
- **Ground truth capture:** because we control the DOM, extract per-word bounding boxes
  via Playwright (`Range.getClientRects()` over text nodes) → JSON sidecar per page:
  `{markdown, plain_text, words:[{text, bbox}], tables:[html], fields:{...}}`.

### 7.3 Degradation pipeline (`data/synth/degrade.py`)
- **augraphy** library: scanner noise, blur, brightness/contrast, JPEG artifacts, paper
  texture, bleed-through, folds, shadows, slight rotation/perspective.
- 3 severity tiers per page (clean / medium / heavy). Keep the clean original too —
  curriculum: train on all tiers, eval per tier.
- Geometric transforms must transform the bboxes identically (affine on coords).

### 7.4 Task/label generation
- **Page → Markdown** (primary task) — from sidecar.
- **Table → HTML/Markdown** — render standalone tables too.
- **Form → JSON** key-value extraction.
- **Grounded OCR**: "read region <bbox>" and "locate text X → bbox" pairs.
- **Page QA** (`data/synth/qa_gen.py`): use a cheap strong LLM (Gemini Flash / GPT-4o-mini /
  Claude Haiku) over the *ground-truth text* (not the image) to write 3–5 QA pairs per
  page + 1–2 **unanswerable** questions. Gold bbox = bbox of sentence containing answer.
  Budget cap: generate QA for a 10–20K page subset only.
- **Abstention labels**: heavy-degradation pages where GT is unreadable → target output
  is the refusal schema; unanswerable QA → target "NOT_IN_DOCUMENT".

### 7.5 Real data mix-in (10–20% of training mix)
- KHATT, Muharaf (handwriting), IFN/ENIT; any permissively-licensed Arabic OCR sets on HF;
  KITAB-Bench *train* portions only if provided as train. License check everything;
  record licenses in `data/LICENSES.md`.

### 7.6 Quality loop
- Render → run base Qwen-VL OCR → flag pages with absurd CER vs GT (often rendering bugs)
  → inspect/fix templates. Manual spot check 1 page per 500.

### 7.7 Deliverables
- `warraq-synth` CLI: `generate --n 20000 --mix config/mix_v1.yaml`.
- Dataset cards + stats notebook (page-type distribution, font histogram, degradation tiers).
- Published later on HF as `warraq-synth-arabic-docs` (Phase 5).

---

## 8. PHASE 2 — Model Training (3–4 weeks)

### 8.1 Base model decision
- **Default: Qwen2.5-VL-3B-Instruct** (recipe proven by SynthDocs paper; native bbox
  grounding support; vLLM-supported; 3B = cheap to train & serve).
- **Check at phase start:** Qwen3-VL small variants (2B/4B/8B) or newer Arabic-strong
  VLMs may have superseded it — re-verify SOTA small VLM + vLLM support + license before
  committing. Decision logged in §17.
- Stretch comparison: 7B variant if 3B plateaus and budget allows.

### 8.2 Method & stack
- **QLoRA first** (4-bit base, LoRA adapters). Framework: **LLaMA-Factory** or **ms-swift**
  (both support Qwen-VL multimodal SFT; pick whichever has smoother Qwen-VL grounding
  format support at the time; Unsloth as memory-efficient alternative for small GPUs).
- Starting hyperparameters (tune from here):
  - LoRA: r=32, alpha=64, dropout=0.05, targets: all attention + MLP projections
    (+ vision-language merger/projector trainable).
  - lr 1e-4 (LoRA) cosine, warmup 3%, epochs 1–2 over mix, eff. batch 64 via grad accum,
    bf16, max sequence ~6–8K tokens (full-page markdown outputs are long).
  - Image resolution: cap Qwen-VL `max_pixels` ≈ 1280–1536 longest side for docs;
    resolution↔memory tradeoff — run an ablation (768 vs 1024 vs 1536).
- **Training mix v1 (by sample count):** 45% page→MD, 15% tables, 15% forms→JSON,
  10% grounded OCR, 10% QA, 5% abstention. Revisit after first eval.
- **Eval-gated training:** after each epoch/checkpoint, auto-run smoke eval suite
  (~100 samples); full suite on best checkpoints. All runs in W&B with config + git SHA.

### 8.3 Compute & cost plan
- **Free tier first:** Kaggle (30 GPU-hrs/wk, T4×2/P100) — enough for 3B QLoRA pilot runs
  at low resolution with Unsloth. Colab/Lightning free credits as backup.
- **Paid runs:** RunPod / Vast.ai — RTX 4090 24GB (~$0.3–0.6/hr range, varies) for QLoRA;
  A100 80GB (~$1–2/hr range) for higher-resolution / 7B runs.
- Estimate: 100–250 GPU-hours total across experiments → **~$60–300**.
- Frontier API spend (baselines + QA gen + judge): **~$50–150** with caps.

### 8.4 Quantization & packaging
- AWQ 4-bit (vLLM-servable) + GGUF (llama.cpp local demo). Re-run full eval suite on the
  quantized model; publish the accuracy delta honestly.
- HF model card with eval table, training config, data description, limitations section
  (handwriting weak, dialect text untested, etc. — honesty = credibility).

### 8.5 Definition of Done
Fine-tuned + quantized model, beats base by wide margin everywhere, beats ≥1 frontier
API overall or all of them on a named slice; everything reproducible from configs.

---

## 9. PHASE 3 — Grounded RAG Product (3–4 weeks)

### 9.1 Ingestion
- Upload PDF/images → raster pages (pymupdf) → Warraq-VLM via vLLM → markdown + word
  bboxes → chunk by structure (headings/paragraph) carrying `{page, bbox[]}` metadata
  → embed (**BGE-M3** primary; ablate vs multilingual-e5-large) → **qdrant** index.
- Optional two-stage mode for dense pages: DocLayout-YOLO / Surya-layout regions →
  per-region VLM OCR. Start single-shot full-page; two-stage is a perf lever, not v1.

### 9.2 Query path
- Embed → top-k retrieve → rerank (bge-reranker-v2-m3) → answer generation:
  - v1 generator: Warraq-VLM itself over retrieved page crops (keeps story pure-open), OR
    a text LLM over extracted markdown — ablate both, report both.
- **Citations:** answer sentences map to source chunks → chunks map to bboxes → frontend
  draws highlight overlays on the page image. Citation = the product's signature move.
- **Abstention gate:** combine retrieval score threshold + answer self-consistency
  (k=3 sampled answers agreement) + model's verbalized confidence → answer / hedge /
  refuse. Operating point chosen from the Phase 0 risk–coverage curve.

### 9.3 Frontend (Next.js, RTL-first)
- Pages: Upload → Document viewer (image + overlay layer) → Chat panel with citations
  (click citation → jump & flash highlight) → confidence indicator (high/low/abstained).
- Arabic-first UI, English toggle. Tailwind. Reuse portfolio design system (GSAP/Framer
  flourishes welcome but performance first).

### 9.4 Definition of Done
60-second demo flow: drag in a scanned Arabic contract → ask 3 questions in Arabic →
2 answered with glowing highlights on the scan, 1 honestly refused. Record this as the
README GIF.

---

## 10. PHASE 4 — Serving & Infra (2 weeks — home turf)

- **Helm chart** `charts/warraq`: api, frontend, vllm (GPU node selector), qdrant,
  embedding-server, otel-collector, prometheus, grafana.
- **Local:** k3d with GPU passthrough if available; CPU fallback = GGUF llama.cpp service
  (slow but demos the architecture). **Cloud demo:** Modal / RunPod serverless for the
  GPU piece (scale-to-zero; free monthly credits), HF Space as fallback demo.
- **GitOps:** FluxCD watching the repo (you already know this cold — make it shine).
- **Observability:** OTel traces ingest→ocr→embed→retrieve→generate; Grafana dashboard:
  p95 latency per stage, tokens/sec, abstention rate, GPU util.
- **CI (GitHub Actions):**
  1. lint+tests; 2. smoke eval (30 samples, CPU/GGUF or small API budget) on PRs touching
  model/prompts/pipeline — **fail the PR if CER regresses >X% or abstention precision
  drops** (eval-gated CI = the Sentinel/QA superpower, productionized);
  3. nightly full eval (manual trigger to control cost).

---

## 11. PHASE 5 — Launch & Credibility (1–2 weeks)

- README: hero GIF, leaderboard, architecture diagram, quickstart (`helm install` +
  `docker compose` path), honest limitations.
- HF releases: model (+AWQ, +GGUF), `warraq-synth` dataset, eval results space.
- **Technical report** (8–12 pp): problem → data engine → training → evals → product.
  PDF via LaTeX/Typst + long-form blog version.
- Distribution: LinkedIn (Arabic + English posts), X, r/LocalLLaMA, HF community,
  Arabic dev communities/Discords, JOSA (Jordan Open Source Association) — present a
  talk if possible. Email the KITAB-Bench authors with results (MBZUAI — they respond
  to good community work; potential collab/visibility).
- CV bullet bank: one quantified line per phase, drafted at launch.

---

## 12. Timeline (part-time, ~10 hrs/week, alongside job + graduation + Sentinel)

| Phase | Duration | Calendar (start mid-June 2026) |
|---|---|---|
| 0 — Eval harness | 2–3 wks | late June → mid July |
| 1 — Data engine | 3–4 wks | mid July → mid Aug *(graduation buffer built in)* |
| 2 — Training | 3–4 wks | mid Aug → mid Sep |
| 3 — Product | 3–4 wks | mid Sep → mid Oct |
| 4 — Infra | 2 wks | mid Oct → end Oct |
| 5 — Launch | 1–2 wks | early Nov 2026 |

Slack built in; phases 3–4 can interleave. If Sentinel pauses, compress by ~1 month.

---

## 13. Budget — tiered (spend is back-loaded: Phases 0–1 ≈ $0; first GPU dollar due Phase 2, ~mid-Aug)

| Tier | What it buys | Est. total (whole project) |
|---|---|---|
| **Floor** — free-tier maximalist | Kaggle 30 GPU-hrs/wk for all training (slower, lower-res); Gemini free tier for most baselines; ~$20–40 of GPT-4o/Claude baseline calls; Modal free credits + HF Spaces for demo | **$50–120** |
| **Comfort** — recommended | Floor + ~100–150 paid GPU-hrs for iteration speed, higher-resolution runs, and 3× variance runs on final numbers (RunPod/Vast 4090 ≈ $0.3–0.5/hr, or Colab Pro+ for 1–2 months) | **$200–350** |
| **Ceiling** | Heavy ablations, 7B variant, A100-class hours | **≤ $500** |

Notes:
- Prepaid hourly credits = hard budget cap (load $25, it cannot spend $26). Safer than subscriptions.
- Colab Pro ($10/mo) is fine for pilots; Pro+ ($50/mo ≈ 35–40 A100-hrs) can cover a lean Phase 2,
  but session disconnects + ephemeral disk add friction vs RunPod/Vast for long runs
  (checkpoint-resume logic required either way).
- API spend control: hard caps in code; Gemini free tier first; 3× variance runs only for
  the final published table, single runs during development.
- Tier decision deferred to Phase 2 start — choose after Phases 0–1 are done and momentum is real.
- Domain ~$12/yr; demo hosting $0–10/mo regardless of tier.

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Fine-tune doesn't beat frontier overall | Own a slice (degraded scans / forms); pivot headline to "matches frontier at 1/50 cost, runs on-prem" — still a killer claim (sovereignty). Publish honest numbers regardless. |
| Synthetic→real domain gap | 10–20% real data mix-in; eval has real public scans; degradation realism via augraphy tuning. |
| Handwriting too hard | Declared stretch goal from day 1; reported as "hard slice", not hidden. |
| Bbox ground truth bugs (RTL/bidi quirks) | Visual QA notebook overlaying boxes on renders; fix templates early. |
| Scope explosion | This file is the contract. New ideas → §17 backlog, not the sprint. |
| Time collision (graduation Aug, Sentinel) | Phase 1 scheduled around graduation; weekly 30-min review ritual; either project may pause but never silently. |
| Eval contamination | Never train on KITAB eval splits; Warraq-Eval pages excluded from generator seeds by hash. |
| License issues | `data/LICENSES.md` ledger; permissive-only for released dataset. |

---

## 15. Workflow & Tooling Contract

- **This chat (Claude):** architecture, design reviews, eval design, unblocking,
  result interpretation. DESIGN.md updates decided here.
- **Claude Code (VS Code):** all implementation, phase by phase, reading this file.
  Prompt pattern: "Read DESIGN.md §6. Implement the eval harness skeleton: …"
- **Source of truth:** this DESIGN.md. Decision log in §17. No undocumented pivots.
- Tracking: W&B (experiments) + GitHub Projects board (phases → issues).

---

## 16. Repository Layout

```
warraq/
├── README.md                 # hero, leaderboard, quickstart, GIF
├── DESIGN.md                 # this file
├── configs/                  # mix_v1.yaml, train_*.yaml, eval suites
├── data/
│   ├── synth/                # render.py, degrade.py, forms.py, qa_gen.py, templates/
│   ├── corpora/              # download scripts only (no large files in git)
│   ├── eval/                 # Warraq-Eval manifests + loaders (KITAB loader here)
│   └── LICENSES.md
├── training/                 # llamafactory/swift configs, train.sh, quantize.py
├── evals/
│   ├── harness.py  metrics.py  arabic_normalize.py  report.py
│   ├── baselines/            # gpt4o.py gemini.py claude.py qwen_base.py tesseract.py ...
│   └── results/              # committed run JSONs
├── serving/
│   ├── api/                  # FastAPI app (ingest, query, abstention gate)
│   └── vllm/                 # launch configs
├── app/                      # Next.js frontend (RTL)
├── infra/
│   ├── charts/warraq/        # Helm
│   ├── flux/                 # GitOps manifests
│   └── otel/  grafana/
├── notebooks/                # data QA, bbox overlay checks, ablation analysis
├── report/                   # technical report (typst/latex)
└── .github/workflows/        # ci.yml (lint+test+smoke-eval), nightly-eval.yml
```

---

## 17. Decision Log & Open Questions

**Decided (v0.1):**
- D1: Eval-first; KITAB-Bench as north-star benchmark. ✅
- D2: Base model Qwen2.5-VL-3B (pending Phase-2 SOTA re-check). ✅
- D3: QLoRA via LLaMA-Factory/ms-swift; W&B tracking. ✅
- D4: BGE-M3 embeddings + qdrant; rerank bge-reranker-v2-m3. ✅
- D5: Synthetic-first data with 10–20% real mix-in; no real personal data ever. ✅
- D6: Codename Warraq; "Baseer" excluded (taken). ✅
- D7: Public repo from day one; Apache 2.0 license (matches Qwen). ✅
- D8: KITAB-Bench is the headline benchmark, loaded from `ahmedheakl/arocrbench_*`
  (corrected path — `mbzuai-oryx/KITAB-Bench` is code-only, not data). ✅
- D9: `Misraj/Misraj-DocOCR` used ONLY as a temporary pipeline bring-up dataset during
  Phase 0 setup (datasets v5 dropped `trust_remote_code`; KITAB path was wrong at first).
  ⚠️ It may be a TRAINING split → contamination risk → MUST NOT appear in any headline
  leaderboard. Retire it from `--suite` once KITAB loaders are wired. ✅

**Open:**
- O1: Final name + domain.
- O2: Qwen3-VL vs Qwen2.5-VL at Phase 2 start (re-verify SOTA + vLLM + license).
- O3: Generator for QA answers in product: Warraq-VLM-only vs hybrid text-LLM (ablate).
- O4: ColQwen/ColPali visual retrieval — v2 backlog.
- O5: Handwriting push (Muharaf fine-tune round) — post-v1 backlog.
- O6: Tashkeel module ("Grammarly for Fusha") as Warraq add-on — backlog.
- O7: Cadence vs Sentinel (alternate weeks? Warraq-primary until Phase 0 done?).

---

## 18. Reference Links

- KITAB-Bench: arxiv.org/abs/2502.14949 · github.com/mbzuai-oryx/KITAB-Bench
- Cross-Lingual SynthDocs (training recipe): arxiv.org/abs/2511.04699
- CAMEL-Bench: arxiv.org/abs/2410.18976
- CATT diacritization (for O6): arxiv.org/abs/2407.03236
- Tools: augraphy · Playwright · LLaMA-Factory · ms-swift · Unsloth · vLLM · qdrant ·
  BGE-M3 · jiwer · DocLayout-YOLO · Surya · pymupdf · W&B
- Data: Arabic Wikipedia · Hindawi · OpenITI · Tashkeela · KHATT · Muharaf · Masader

---

*v0.1 — drafted June 11, 2026. Owner: Mousa. Architect-on-call: Claude.*
