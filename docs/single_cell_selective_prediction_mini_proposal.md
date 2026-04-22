# Mini-Proposal: Selective Prediction for Single-Cell Annotation under Biological Shift

## Working title
**Selective Prediction for Single-Cell Annotation under Biological Shift: Why Marginal Reliability Is Not Enough**

## One-paragraph pitch
Automated single-cell annotation is now routinely performed by transferring labels from a reference atlas to a new query dataset. However, in practice, failures are concentrated in precisely the regimes that matter most biologically: new donors, new protocols, batch effects, disease-specific states, and rare or unseen cell populations. Recent work has brought conformal prediction and uncertainty quantification to single-cell annotation, and recent evaluations show that out-of-distribution (OOD) detection can help identify novel cell types and shifts. But the dominant reliability guarantees remain **marginal**: they average over all cells, even though the hardest and most consequential errors are concentrated in specific subgroups. This proposal studies **selective prediction** for single-cell annotation under biological shift, with the central thesis that **marginal reliability is insufficient** and that the rejector must be **subgroup-aware** and **shift-aware**. The goal is a paper whose contribution is not a large framework, but a clean combination of: (i) a formal failure analysis of marginal selective reliability under subgroup imbalance and biological shift, (ii) a lightweight abstention method, and (iii) controlled experiments on realistic single-cell reference-mapping settings.

## Why this direction fits NeurIPS
NeurIPS 2026 explicitly lists **AI/ML for health and biotechnology**, **probabilistic methods**, and **theory**, and also encourages **in-depth analysis of existing methods that provide new insights into their limitations or behavior**. This project sits exactly at that intersection: it is biologically grounded, centered on uncertainty-aware selective prediction, and driven by a precise limitation of current reliability guarantees.

## Problem setup
We consider supervised reference-to-query cell-type annotation. A model is trained on a labeled reference datasetd
\[
\mathcal{D}_{\mathrm{ref}} = \{(x_i, y_i, b_i)\}_{i=1}^n,
\]
where \(x_i\) is a cell, \(y_i \in \mathcal{Y}\) its cell-type label, and \(b_i\) a metadata-defined subgroup such as batch, donor, protocol, tissue, or disease status. At test time, a query dataset contains cells
\[
\mathcal{D}_{\mathrm{qry}} = \{(x_j, y_j, b_j)\}_{j=1}^m,
\]
possibly with biological or technical shift, including unseen cell types.

A base annotator produces class probabilities \(p_\theta(y\mid x)\). A selective predictor augments this with an acceptance score \(s(x)\) and predicts only when \(s(x) \ge \tau\); otherwise it abstains. The selective risk at threshold \(\tau\) is
\[
R(\tau) = \mathbb{E}[\ell(\hat y(X), Y) \mid s(X) \ge \tau],
\]
and the coverage is
\[
C(\tau) = \mathbb{P}(s(X) \ge \tau).
\]
The standard goal is to maximize coverage subject to a risk target \(R(\tau) \le \alpha\).

The key issue is that the standard guarantee is **marginal** over all query cells. In biological practice, however, the relevant failure modes are subgroup-structured:
- rare cell populations,
- donor-specific shifts,
- protocol/platform mismatch,
- disease-state shifts,
- and truly unseen cell types.

The central research question is therefore:

> **Can we design a selective annotation rule that remains reliable under biologically meaningful subgroup and shift structure, rather than only on average across all cells?**

## Core hypothesis
A selective predictor calibrated only for marginal risk can still make systematically harmful errors on underrepresented or shifted biological subgroups. A better abstention rule should combine **confidence**, **local support**, and **shift-awareness**, and its theoretical guarantee should reflect the subgroup structure of the deployment setting.

## Gap in current literature
This direction is timely for three reasons:
1. Recent work has introduced **conformal inference for reliable single-cell annotation**, including OOD cell-type detection and calibrated prediction sets.
2. Separate recent work evaluates **OOD detection for automatic single-cell annotation** and shows that novel cell types and natural biological/technical shifts materially affect reliability.
3. Recent benchmarking on reference mapping and integration emphasizes that good global integration quality does not automatically translate to robust query mapping or reliable unseen-population behavior.

What is still missing is a theory-and-method paper that connects these threads through **selective prediction**: not just uncertainty sets, but explicit predict-or-abstain decisions with guarantees and analysis under subgroup/shift structure.

## Technical objective
The paper will aim for three linked contributions.

### Contribution 1: A failure theorem for marginal selective reliability
We will formalize the fact that a selective rule can satisfy a marginal risk target while still exhibiting arbitrarily poor subgroup risk on a small but important subgroup. A target statement is:

**Proposition (Failure of marginal selective control).** There exist mixtures of biological subgroups \(G \in \{1,\dots,K\}\) and score distributions such that a threshold rule \(s(x) \ge \tau\) satisfies the marginal selective risk target \(R(\tau) \le \alpha\), yet for some subgroup \(g\),
\[
R_g(\tau) = \mathbb{E}[\ell(\hat y(X),Y) \mid s(X) \ge \tau, G=g]
\]
can be arbitrarily larger than \(\alpha\).

This proposition gives the paper a sharp opening: **current marginal guarantees do not align with biological deployment risk**.

### Contribution 2: A subgroup-/shift-aware acceptance score
We will define a lightweight score
\[
s(x) = \lambda_1 \cdot \text{Conf}(x) + \lambda_2 \cdot \text{Support}(x) - \lambda_3 \cdot \text{Shift}(x),
\]
where:
- **Conf** is a standard confidence term, such as max softmax probability, logit margin, or nonconformity inverse;
- **Support** measures local support of the query cell in the reference embedding, such as kNN density or neighborhood agreement;
- **Shift** measures mismatch between the query cell and the calibration/reference distribution, for example domain discriminator score, energy score, density ratio surrogate, or nearest-neighbor discrepancy.

This remains intentionally small. The novelty is not an architecture, but the decision-theoretic combination of these terms and the theoretical interpretation.

### Contribution 3: A bound linking subgroup risk to shift discrepancy
A realistic target theorem is not exact conditional coverage, which is often too strong. Instead, we aim for an approximate result of the form:
\[
R_g(\tau) \le R_{\mathrm{marg}}(\tau) + \Delta_g,
\]
where \(\Delta_g\) depends on subgroup-specific shift or calibration mismatch, such as:
- total variation / Wasserstein discrepancy between subgroup-conditioned score distributions,
- density-ratio misalignment,
- or calibration-sample scarcity.

This theorem is enough to support the message that **group-aware abstention is justified when subgroup-specific score behavior diverges from the marginal calibration population**.

## Method plan
The method should stay simple enough to execute quickly.

### Backbone
Use an existing annotation pipeline rather than inventing a new one. Options:
- scVI/scANVI latent space + classifier,
- a strong reference-mapping baseline such as a standard atlas transfer pipeline,
- or a simple MLP / linear classifier on integrated embeddings.

The paper should be **backbone-agnostic** in presentation.

### Acceptance score families
We will compare a small family of rejectors:
1. **Confidence-only**: max probability, entropy, margin.
2. **Conformal-only**: thresholding based on nonconformity / prediction-set size.
3. **Support-aware**: density or kNN agreement in embedding space.
4. **Shift-aware**: score with explicit domain-mismatch term.
5. **Combined selective score (ours)**: confidence + support − shift.

### Calibration strategy
There are two good options.

**Option A (faster):** split calibration on a held-out validation set and choose \(\tau\) to satisfy a target empirical risk bound.

**Option B (stronger):** weighted or stratified calibration by subgroup proxies, giving a more direct path to the subgroup bound.

The recommended plan is to start with Option A and add Option B only if the experiments show a meaningful gap.

## Experimental plan
The empirical section must be compact and disciplined.

### Main settings
We will evaluate under four stress regimes:
1. **In-distribution reference mapping**: same tissue/task, mild batch shift.
2. **Cross-batch / cross-donor** shift.
3. **Cross-platform / protocol** shift.
4. **Novel cell types / held-out labels** in the query set.

### Dataset construction strategy
Rather than relying on one giant benchmark, we should construct 3–4 controlled settings from public scRNA-seq references/query datasets. The crucial point is not the number of datasets but the clarity of the shift taxonomy.

For each setting, define:
- the label ontology,
- the subgroup variable(s),
- the train/calibration/test split,
- and whether the shift is biological, technical, or both.

### Metrics
The main metrics should be:
- **Selective risk–coverage curve**,
- **Coverage at fixed risk target**,
- **Risk at fixed coverage**,
- **Worst-group selective risk**,
- **Coverage parity / disparity across subgroups**,
- **OOD detection AUROC/AUPRC** for unseen cell types,
- and optionally calibration error of the acceptance score.

The paper should report both global and subgroup metrics; otherwise the central claim is not substantiated.

### Baselines
Keep the baseline set tight:
- confidence thresholding,
- entropy thresholding,
- conformal prediction baseline for reliable annotation,
- standard OOD score baseline (energy or ensemble-based if affordable),
- support-only / density-only rejector,
- and the proposed combined score.

### Ablations
Essential ablations:
1. Remove support term.
2. Remove shift term.
3. Use oracle subgroup labels only at calibration.
4. Weighted vs unweighted calibration.
5. Rare-group stress test by downsampling calibration cells from a subgroup.

## Figure plan
The paper can be carried by four figures.

### Figure 1: Problem schematic
Reference atlas → query cells under biological shift → annotator + accept/reject rule → accepted predictions and abstentions. The figure should explicitly illustrate that marginal reliability can hide subgroup failure.

### Figure 2: Counterexample / theory illustration
A simple 2-group toy plot showing that marginal risk is controlled while a rare shifted subgroup has large selective risk.

### Figure 3: Main experimental result
Risk–coverage curves for confidence, conformal, and subgroup-/shift-aware selective scores across one representative cross-batch or novel-cell-type setup.

### Figure 4: Subgroup analysis
Bar or line plot of selective risk and coverage across batches/donors/cell groups at a fixed operating point.

## Table plan
Two tables are enough.

### Table 1: Main benchmark summary
For each experimental setting: coverage at target risk, worst-group risk, AUROC for unseen-cell detection, and abstention rate.

### Table 2: Ablation summary
Effect of support term, shift term, and weighted calibration on worst-group selective risk and average coverage.

## What makes this proposal strong
This proposal has four concrete strengths.

1. **Domain relevance is immediate.** Single-cell annotation is a core bioinformatics task, and failures under donor/protocol/novel-state shift are biologically meaningful rather than synthetic.
2. **The ML core is sharp.** This is fundamentally a selective prediction paper, not a bioinformatics pipeline paper.
3. **The theorem target is realistic.** Exact subgroup guarantees are hard, but a failure result plus an approximate subgroup-risk bound is very plausible.
4. **Execution is fast.** The project does not require inventing a new foundation model or building a large systems stack.

## Main risks and how to control them
### Risk 1: The theorem becomes too weak
Mitigation: make the negative result central. Even if the positive theorem is modest, the paper remains valuable if the empirical story strongly validates the failure mode.

### Risk 2: The method looks too simple
Mitigation: lean into it. The paper should explicitly argue that the contribution is a **decision-theoretic correction** to current uncertainty practice, not a bigger model.

### Risk 3: Benchmark sprawl
Mitigation: use only 3–4 carefully designed settings with clear subgroup semantics. Avoid turning the paper into a broad benchmark.

### Risk 4: Reviewers ask whether this is “just fairness by another name”
Mitigation: position the work around **deployment reliability under biological shift**, where subgroup structure is induced by real assay and biological variation rather than only demographic partitioning.

## 6-week execution roadmap
### Week 1 — Benchmark and backbone setup
- Select 3–4 public reference/query settings.
- Standardize preprocessing and subgroup metadata.
- Train or reuse one strong base annotator.
- Implement confidence and entropy rejectors.

**Deliverable:** first selective risk–coverage curves on at least one shift setting.

### Week 2 — Conformal and support-aware baselines
- Add conformal baseline for reliable annotation.
- Add support-aware score (kNN density / agreement).
- Run initial rare-group and novel-cell experiments.

**Deliverable:** baseline comparison table.

### Week 3 — Failure analysis and theorem draft
- Construct two-group toy model and prove marginal-failure proposition.
- Quantify subgroup failure empirically on real data.
- Draft theorem statements and notation.

**Deliverable:** Figure 2 + theorem skeleton.

### Week 4 — Shift-aware score and weighted calibration
- Add explicit shift term.
- Add weighted or subgroup-aware calibration variant.
- Run cross-batch/cross-donor main comparisons.

**Deliverable:** main experimental figure and first full results table.

### Week 5 — Ablations and paper drafting
- Finish ablations.
- Write introduction, related work, and method.
- Tighten theory wording to match the achieved result.

**Deliverable:** full draft v1.

### Week 6 — Finalization
- Polish figures and tables.
- Rewrite abstract and discussion around the sharpest claim actually supported by the evidence.
- Trim scope if needed.

**Deliverable:** submission-ready draft.

## Recommended positioning sentence for the abstract
We study selective prediction for single-cell annotation under biological shift and show that marginal reliability guarantees can mask severe subgroup-specific failures under batch, donor, protocol, and novel-cell-type shift. We propose a lightweight subgroup- and shift-aware abstention rule, establish a failure result for marginal selective control and an approximate subgroup-risk bound, and demonstrate improved worst-group reliability at competitive coverage on realistic single-cell reference-mapping settings.

## Final recommendation
If the deadline is tight, this is the right project to prioritize. It is the best balance among novelty, theorem feasibility, domain relevance, and implementation speed. The paper should be written as a **theory-plus-analysis selective prediction paper in a biologically important setting**, not as a new large single-cell framework.

