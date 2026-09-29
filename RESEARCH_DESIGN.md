# AfriARAG Research Design — Core Protocol (Sections 1–69)

## 1–8. Study definition

**Title:** *AfriARAG: When and Where to Retrieve? Adaptive Cross-Lingual Hybrid Retrieval for Low-Resource African Sentiment Analysis.*

**Problem:** fixed RAG assumes the same retrieval configuration is appropriate for every input. African-language sentiment inputs vary in language, code-mixing, lexical rarity, resource availability and classifier uncertainty.

**Aim:** design and evaluate a query-aware controller that determines whether to retrieve, which retrieval family to use, which language evidence pool to search, and how much evidence to retrieve.

**Objectives:** quantify fixed-retrieval limitations; implement sparse/dense/hybrid retrieval; learn adaptive routing; test strict zero-shot Oromo/Tigrinya routing; evaluate effectiveness and efficiency.

**RQ1–RQ5:** retrieval benefit; policy variation; controller vs fixed; strict zero-shot cross-lingual routing; observable features associated with retrieval help/harm.

**Hypotheses:** retrieval is selectively beneficial; one fixed policy is not globally optimal; learned routing can recover oracle headroom; adaptive retrieval can reduce unnecessary retrieval; cross-lingual evidence can help strict zero-shot targets.

## 9–13. Data and formalization

Primary dataset: AfriSenti / SemEval 2023. Supervised languages are `amh arq ary hau ibo kin pcm por swa tso twi yor`; strict zero-shot targets are `orm tir`. Labels are positive, neutral and negative.

A query is `q_i=(x_i,l_i,y_i)`. A retrieval action is `a(q)=(g,r,s,k,alpha)`, where `g` is retrieve/no-retrieve, `r` is BM25/dense/hybrid, `s` is same/multilingual/cross, `k` is retrieval depth and `alpha` is hybrid dense weight.

**Strict zero-shot rule:** Oromo/Tigrinya labels cannot train or tune the classifier, retriever, controller, policy set, k, alpha, language routing or prompts. They are used only for final scoring.

## 14–18. Retrieval architecture

Only training examples may be indexed. Maintain same-language and combined multilingual pools. Cross-lingual pools exclude the target language. BM25 provides lexical evidence; a multilingual sentence encoder provides semantic evidence; hybrid fusion uses normalized scores:

`S_h(q,d)=alpha*S_dense(q,d)+(1-alpha)*S_BM25(q,d)`.

Stage 1 freezes `alpha=.5` for hybrid policies and uses `k={3,5}`. Expand to `k={1,3,5,10}` and `alpha={.25,.5,.75}` only after oracle analysis supports the main hypothesis.

## 19–25. Oracle and controller

For every development query evaluate every candidate policy. Let `C_i` be policies that predict the gold label correctly. The oracle selects the lowest-cost member of `C_i`; if `C_i` is empty, mark the query `unresolved`.

Controller features may include language, length, script/code-mixing proxies, punctuation, emoji, lexical rarity, shallow retrieval signals and no-RAG uncertainty. Gold-label-derived information is forbidden at inference.

Stage 1 uses an interpretable tree ensemble. A later cost-aware utility may optimize `performance - lambda_cost*cost - lambda_latency*latency`.

## 26–29. Classifier layers and baselines

Use two layers. **Layer A** is a cheap deterministic classifier for large retrieval/oracle experiments. **Layer B** is the publication LLM-RAG classifier after the oracle gate.

Core baselines: class prior, fine-tuned multilingual/African encoder, LLM no-RAG, static few-shot, random demonstrations, fixed BM25, fixed dense, fixed same-language hybrid, fixed multilingual hybrid, fixed cross-lingual, adaptive AfriARAG and oracle AfriARAG.

The primary comparison is adaptive AfriARAG vs the strongest fixed retrieval policy. The oracle is an upper bound, not a deployable method.

## 30–36. Experiment stages

1. Audit/freeze data and detect duplicates/leakage.
2. Run fixed retrieval baselines.
3. Generate the development oracle and analyze policy distribution.
4. Train the controller on supervised-language development data only.
5. Freeze all choices and evaluate held-out supervised test data.
6. Evaluate strict zero-shot Oromo/Tigrinya using supervised-language evidence only.
7. Analyze cross-lingual source routing.

Do not proceed to large paid LLM runs until Stage 3 demonstrates meaningful oracle headroom and policy diversity.

## 37–47. Metrics and statistics

Primary metric: **macro-F1**. Also report weighted-F1, accuracy, macro precision/recall, per-class F1 and macro-language F1.

Retrieval metrics: retrieval rate, average k, same-/cross-language rate, latency, top-k label purity for analysis only, tokens/query, cost/query and F1 gain per 1,000 extra tokens.

If valid probabilities exist, report ECE/Brier. Use paired bootstrap confidence intervals for adaptive-vs-best-fixed differences and multiple random seeds for learned models.

Ablate selective retrieval, sparse retrieval, dense retrieval, adaptive alpha, dynamic k, language routing, uncertainty, retrieval signals and code-mixing features.

Measure retrieval harm (no-RAG correct -> RAG wrong), retrieval rescue (no-RAG wrong -> RAG correct), policy accuracy, oracle-compatible rate and controller regret.

## 48–55. Analysis and validity

Analyze results by query difficulty, language, source-language routing, sentiment class and qualitative error bucket: sarcasm, negation, code mixing, idiom, implicit sentiment, named entities, dialect spelling, irrelevant retrieval and contradictory demonstrations.

Annotator disagreement is optional rather than core unless individual annotations are reliably available in the frozen release.

Mandatory leakage controls: no test item in indices; no validation/test labels in prompts; duplicate and near-duplicate checks; development-only tuning; choices frozen before test; Oromo/Tigrinya labels never used for tuning.

Every experiment should log experiment ID, git commit, dataset checksum, seed, language, split, retriever, embedding model, classifier, controller, k, alpha, scope, prompt version, predictions, gold labels, retrieval IDs/scores, latency, tokens and cost.

## 56–62. Configuration and hypothesis tests

Experiments are YAML-controlled. Model selection follows **Train -> Development -> Freeze -> Test**.

The main matrix compares no retrieval, BM25 same, dense same, hybrid same, multilingual, cross-lingual, adaptive and oracle.

Primary hypothesis test: adaptive macro-F1 vs strongest fixed macro-F1. Secondary tests examine cost/latency and strict zero-shot performance.

## 63–69. Contributions, reporting and execution

If supported, contributions are: a joint adaptive retrieval architecture; retrieval help/harm analysis across African languages; strict zero-shot cross-lingual routing; a retrieval-oracle/policy-learning framework; and effectiveness-efficiency evaluation.

Expected figures: architecture, oracle action distribution, adaptive-vs-fixed performance, F1-cost frontier, cross-language source heatmap and benefit by query difficulty.

Expected tables: dataset statistics, baselines, fixed retrieval, adaptive/oracle, zero-shot, ablations, efficiency and error analysis.

Paper structure: Introduction; Related Work; Methodology; Experimental Setup; Results; Analysis; Discussion; Conclusion.

Execution order: audit -> BM25 -> dense -> hybrid -> cheap fixed experiments -> oracle -> oracle analysis -> features -> controller -> supervised adaptive -> strict zero-shot -> publication LLM layer -> ablations -> bootstrap tests -> qualitative analysis -> freeze tables/figures -> manuscript.

**Falsification criterion:** if one fixed policy nearly matches the oracle for almost all queries/languages, the case for adaptation is weak and the action space or hypothesis should be revised before expensive experiments.
