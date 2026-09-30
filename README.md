# AfriARAG

**Adaptive Cross-Lingual Hybrid Retrieval for Low-Resource African Sentiment Analysis**

AfriARAG studies whether retrieval decisions for African-language sentiment classification should be made per query rather than fixed globally.

The core policy is:

```
a(q) = {retrieve?, retriever, scope, k, alpha}
```

where the retriever is BM25, dense, or hybrid; scope is same-language, multilingual, or cross-lingual; `k` is retrieval depth; and `alpha` controls dense-vs-sparse fusion.

## Stage 1 scope

The repository currently implements the initial Sections 1–69 research scope:

- AfriSenti data acquisition and audit
- leakage-safe split preparation
- no-retrieval baseline
- BM25, dense, and hybrid retrieval
- same-language, multilingual, and cross-lingual retrieval
- fixed-policy evaluation
- development-set retrieval oracle
- adaptive controller
- strict zero-shot Oromo/Tigrinya evaluation
- effectiveness, efficiency, retrieval-harm, and significance analysis

## Languages

Supervised: `amh arq ary hau ibo kin pcm por swa tso twi yor`

Strict zero-shot: `orm tir`

## First run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python scripts/download_data.py
python scripts/audit_data.py
python scripts/prepare_leakage_safe_data.py
```

Dense retrieval uses PyTorch through Sentence Transformers. NumPy is kept below 2.0. Requirements select PyTorch 2.2.2 and compatible Transformers 4.x on Intel macOS (x86_64), where newer PyTorch wheels are unavailable. Other platforms use PyTorch 2.5 or newer.

If you are using an existing Intel Mac environment, repair it with:

```bash
python -m pip install --upgrade "numpy>=1.26,<2.0" "torch==2.2.2" "transformers>=4.49,<5" "sentence-transformers>=3.4,<4"
python -m pip check
python -c "import numpy, torch, transformers, sentence_transformers; print('numpy', numpy.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
```

Then rerun the experiment command. Use `python -m pip` while the project `.venv` is active so packages install into the same environment used to run the scripts.

The raw official files remain in `data/afrisenti/`. Primary experiments use the leakage-safe view in `data/afrisenti_clean/`.

Before the full experiment, run a small Hausa smoke test:

```bash
python scripts/run_fixed.py \
  --config configs/stage1.yaml \
  --split dev \
  --languages hau \
  --limit 200 \
  --policies no_retrieval bm25_same_k3 bm25_same_k5 dense_same_k3 dense_same_k5 hybrid_same_k3 hybrid_same_k5
```

This smoke run intentionally excludes multilingual/cross-lingual policies so a CPU-only laptop does not need to embed the entire supervised corpus merely to validate the pipeline. Its outputs are tagged `dev_hau_n200_*.csv`.

If successful, use a GPU-capable machine for the full supervised development experiment:

```bash
python scripts/run_fixed.py --config configs/stage1.yaml --split dev
python scripts/build_oracle.py --input outputs/fixed/dev_predictions.csv --tag dev
python scripts/train_controller.py --predictions outputs/fixed/dev_predictions.csv
```

Stage 1 has now completed its frozen held-out evaluation. The final interpretation
is recorded in `docs/FINAL_RESULTS_STAGE1.md`.

## Publication-analysis package

All post-freeze publication analyses are generated from frozen outputs. They do
not retrain or retune the Stage-1 system.

Run the development-only explanatory ablations first:

```bash
python scripts/run_publication_ablations.py
```

Then build the final publication package:

```bash
python scripts/build_publication_package.py
```

Outputs are written to `outputs/publication/`, including:

- main/fixed-policy/per-language/per-class tables;
- development-only controller feature and action-space ablations;
- rescue/harm summaries and qualitative examples;
- supervised and zero-shot confusion matrices;
- per-language delta and fixed-policy comparison figures;
- efficiency accounting with explicit latency caveats;
- a publication manifest tying the package back to the pre-test freeze.

The held-out test outputs must not be used to tune the frozen Stage-1 models.
Any new controller architecture is a separately named Stage-2 experiment.

See `RESEARCH_DESIGN.md`, `EXPERIMENTS.md`, and
`docs/FINAL_RESULTS_STAGE1.md` for the protocol and final Stage-1 conclusions.
