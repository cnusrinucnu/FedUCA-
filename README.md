# FedUCA — Reproducibility Package

**Title:** FedUCA: Prototype-Guided Federated Learning with Uncertainty  
Calibration for Privacy-Preserving Skin Lesion Classification

**Author:** Srinivas Arukonda, SRM University-AP  
**Email:** srinivas.a@srmap.edu.in  
**ORCID:** 0000-0002-8974-844X  
**Platform:** Kaggle (all experiments run on Kaggle cloud — free)

---

## Overview

This package contains all Jupyter notebooks required to fully reproduce
every experiment in the paper. All experiments run on Kaggle using free
GPU (NVIDIA T4) and free CPU sessions. No local GPU required.

**Total GPU time to reproduce all experiments: ~15 hours**  
**All datasets are free and publicly available on Kaggle**

---

## Step 0 — One-Time Setup

1. Create a free account at kaggle.com
2. Go to Settings → Phone verification (required for GPU access)
3. All notebooks below are uploaded directly to Kaggle

---

## Datasets — What To Add and Where

Every notebook lists which datasets to add. Summary:

| Dataset | Kaggle URL | Size | Used In |
|---------|-----------|------|---------|
| HAM10000 images | kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 | 1.8 GB | Notebook 1 |
| ISIC 2019 images | kaggle.com/datasets/andrewmvd/isic-2019 | 10 GB | Notebook 9 |

All other datasets are **outputs you save yourself** as you run each notebook.
The README for each notebook tells you exactly what to save and what to name it.

---

## Execution Order — Run Notebooks in This Exact Order

```
PHASE 1 → PHASE 2 → PHASE 3 → PHASE 4 → PHASE 5
   ↓          ↓         ↓         ↓         ↓
Embeddings  FedAvg    PGA     FedProto  Calibration
(required   (requires (requires (requires (requires
 first)     Phase 1)  Phase 2)  Phase 1)  Phase 2)

PHASE 6 → PHASE 7 → PHASE 8 → PHASE 9 → PHASE 10
   ↓          ↓         ↓         ↓          ↓
Scalability  5-fold   ISIC     ISIC PGA   ISIC
K=10,20      CV       FedAvg              FedProto

PHASE 11 → PHASE 12
    ↓           ↓
Centralised  Figures +
Baseline     Tables
```

---

## Notebook 1 — HAM10000 Embedding Extraction
**File:** `01_embedding_cache_ham10000.ipynb`  
**Datasets to add:** `kmader/skin-cancer-mnist-ham10000`  
**Accelerator:** GPU T4  
**Runtime:** ~30 min  
**Output — save as dataset:** `ham10000-vit-embeddings-orig`  

Extracts frozen ViT-B/16 SWAG class-token embeddings (768-dim) for all
HAM10000 images. Applies lesion-disjoint GroupShuffleSplit (seed=42):
Train 7,984 / Val 1,005 / Test 1,026. No lesion overlap between splits.

---

## Notebook 2 — FedAvg Baseline (Head Ablation)
**File:** `02_fedavg_head_ablation.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig` (from Notebook 1)  
**Accelerator:** GPU T4  
**Runtime:** ~3 hours  
**Output — save as dataset:** `feduca-phase2-results`  

Runs FedAvg with 4 head architectures × 3 Dirichlet alpha levels × 5 seeds.
K=4 clients, T=60 rounds, E=5 local epochs, Adam lr=5e-3.
Produces Tables 1 and 2 of the paper.

Expected key result:
- Bottleneck N/4, α=1000: F1 = 0.643 ± 0.010
- Bottleneck N/4, α=0.1:  F1 = 0.363 ± 0.116

---

## Notebook 3 — Prototype-Guided Aggregation (PGA)
**File:** `03_pga.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig` + `feduca-phase2-results`  
**Accelerator:** GPU T4  
**Runtime:** ~90 min (5 seeds) + re-run for seeds 5-9  
**Output — save as dataset:** `feduca-phase3-results`  

Runs PGA with Bottleneck N/4, 3 alpha levels.
For the primary α=0.1 comparison, run with seeds 0-4 first,
then seeds 5-9 separately (see Notebook 3b).
Produces Table 3 (primary result) of the paper.

Expected key result:
- PGA, α=0.1, 10 seeds: F1 = 0.483 ± 0.090, p=0.001 vs FedAvg

---

## Notebook 3b — PGA Extended Seeds (Seeds 5-9)
**File:** `03b_pga_seeds5to9.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig`  
**Accelerator:** GPU T4  
**Runtime:** ~45 min  
**Output — save as dataset:** `feduca-extra-a2-results`  

Runs PGA seeds 5-9 at α=0.1 only. Combined with Notebook 3 seeds 0-4
gives the 10-seed primary comparison.

---

## Notebook 4 — FedProx Baseline
**File:** `04_fedprox.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig`  
**Accelerator:** GPU T4  
**Runtime:** ~90 min  
**Output — save as dataset:** `feduca-extra-a-results`  

Runs FedProx (μ=0.01) with Bottleneck N/4, 3 alpha × 5 seeds.
Produces FedProx row in Table 3.

Expected key result:
- FedProx, α=0.1: F1 = 0.414 ± 0.109, p=0.874 vs FedAvg (not significant)

---

## Notebook 5 — FedProto Baseline (HAM10000)
**File:** `05_fedproto_ham10000.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig`  
**Accelerator:** GPU T4  
**Runtime:** ~90 min  
**Output — save as dataset:** `feduca-fedproto-results`  

Runs FedProto (Tan et al., AAAI 2022) with Bottleneck N/4, 3 alpha × 5 seeds.
Key finding: FedProto performs WORSE than FedAvg under α=0.1.

Expected key result:
- FedProto, α=0.1: F1 = 0.365 ± 0.100 (worse than FedAvg 0.409)

---

## Notebook 6 — Federated Temperature Scaling (Calibration)
**File:** `06_calibration.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig` + `feduca-phase2-results`  
**Accelerator:** CPU (no GPU needed)  
**Runtime:** ~30 min  
**Output — save as dataset:** `feduca-phase5-results`  

Computes federated temperature scaling. Each client minimises ECE locally;
server aggregates T* proportionally by validation size.
Produces Table 7 of the paper.

Expected key result:
- ECE: 0.1660 → 0.0663 (-60.0%, p<0.001) at α=1000

---

## Notebook 7 — Scalability (K=10, K=20)
**File:** `07_scalability.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig`  
**Accelerator:** GPU T4  
**Runtime:** ~2 hours  
**Output — save as dataset:** `feduca-extra-a3-results`  

Runs FedAvg and PGA with K=10 and K=20 clients, 3 alpha × 5 seeds.
Produces Table 4 and Figure 7 of the paper.

---

## Notebook 8 — 5-Fold Cross-Validation
**File:** `08_cross_validation.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig`  
**Accelerator:** GPU T4  
**Runtime:** ~90 min  
**Output — save as dataset:** `feduca-cv-results`  

Runs 5-fold lesion-grouped GroupKFold, FedAvg and PGA, α=1000.
Produces Table 5 of the paper.

---

## Notebook 9 — ISIC 2019 Embeddings + FedAvg + PGA
**File:** `09_isic2019_fedavg_pga.ipynb`  
**Datasets to add:** `andrewmvd/isic-2019`  
**Accelerator:** GPU T4  
**Runtime:** ~3 hours (embedding extraction ~1h + runs ~2h)  
**Output — save as dataset:** `feduca-isic-results`  

Extracts ISIC 2019 embeddings and runs FedAvg + PGA.
Produces Table 6 (partial) of the paper.

Expected key result:
- PGA, α=0.1: F1 = 0.524 ± 0.058 vs FedAvg 0.403 ± 0.100

---

## Notebook 10 — ISIC 2019 FedProto
**File:** `10_isic2019_fedproto.ipynb`  
**Datasets to add:** `andrewmvd/isic-2019`  
**Accelerator:** GPU T4  
**Runtime:** ~90 min  
**Output — save as dataset:** `feduca-isic-fedproto-results`  

Runs FedProto on ISIC 2019, 3 alpha × 5 seeds.
Completes Table 6 of the paper.

Expected key result:
- FedProto, α=0.1: F1 = 0.413 ± 0.111

---

## Notebook 11 — Centralised Baseline
**File:** `11_centralised_baseline.ipynb`  
**Datasets to add:** `ham10000-vit-embeddings-orig`  
**Accelerator:** CPU (no GPU needed)  
**Runtime:** ~10 min  
**Output:** Prints results only (no dataset needed)  

Trains Bottleneck N/4 on all training data with no federation.
Establishes the upper bound for comparison.

Expected key result:
- Centralised: F1 = 0.639 ± 0.010

---

## Notebook 12 — All Figures and Tables
**File:** `12_figures_tables.ipynb`  
**Datasets to add:** All result datasets from Notebooks 2-11  
**Accelerator:** CPU  
**Runtime:** ~5 min  

Generates all 9 publication-quality figures (PDF + PNG, 300 DPI)
and all 7 LaTeX tables from the saved result JSON files.

---

## Expected Final Results Summary

| Method | α=1000 | α=1 | α=0.1 |
|--------|--------|-----|--------|
| Centralised (upper bound) | 0.639±0.010 | — | — |
| FedAvg | 0.643±0.010 | 0.627±0.011 | 0.409±0.117 |
| FedProx | 0.607±0.006 | 0.614±0.022 | 0.414±0.109 |
| FedProto | 0.626±0.010 | 0.629±0.014 | 0.365±0.100 |
| **PGA (Ours)** | **0.639±0.013** | **0.631±0.014** | **0.483±0.090** |

| Metric | Value | p-value |
|--------|-------|---------|
| PGA vs FedAvg Δ (α=0.1, 10 seeds) | +0.074 | 0.001 |
| PGA vs FedProto Δ (α=0.1) | +0.118 | 0.021 |
| ECE reduction (federated T*) | -60.0% | <0.001 |

---

## Environment

All notebooks were developed and tested on:
- Platform: Kaggle (free tier)
- GPU: NVIDIA T4 (15 GB VRAM)
- Python: 3.10
- PyTorch: 2.5.1
- torchvision: 0.20.1
- scikit-learn: 1.3.0
- numpy: 1.26.0

No additional installation required on Kaggle — all packages are pre-installed.

---

## Contact

For questions about reproducibility, contact:  
**Srinivas Arukonda** — srinivas.a@srmap.edu.in  
Department of Computer Science and Engineering  
SRM University-AP, Amaravati, Andhra Pradesh, India
