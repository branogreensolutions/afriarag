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
pip install -r requirements.txt

python scripts/download_data.py
python scripts/audit_data.py
python scripts/prepare_leakage_safe_data.py
```

The raw official files remain in `data/afrisenti/`. Primary experiments use the leakage-safe view in `data/afrisenti_clean/`.

Before the full experiment, run a small Hausa smoke test:

```bash
python scripts/run_fixed.py --config configs/stage1.yaml --split dev --languages hau --limit 200
```

If successful, run the full supervised development experiment:

```bash
python scripts/run_fixed.py --config configs/stage1.yaml --split dev
python scripts/build_oracle.py --input outputs/fixed/dev_predictions.csv
python scripts/train_controller.py --oracle outputs/oracle/oracle_labels.csv
```

Do **not** run the frozen supervised test evaluation until the development-stage policy/controller choices have been reviewed and frozen.

See `RESEARCH_DESIGN.md` and `EXPERIMENTS.md` for the protocol.
