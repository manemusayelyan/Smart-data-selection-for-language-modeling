# Capstone: Smart data selection for language modeling

LLMs have achieved remarkable progress through the scaling paradigm: increasing model size, training data, and compute consistently improves performance. However, standard pretraining pipelines still rely on a simple assumption — every token in the training corpus contributes equally to learning. In practice, large-scale web datasets are highly heterogeneous: many tokens are repetitive and easily predictable, while others contain rare, complex, or domain-specific information that provides significantly stronger learning signals. Treating all tokens uniformly leads to substantial computational inefficiency, especially at the scale of billions of training tokens.

At the same time, constructing high-quality pretraining datasets has become increasingly difficult. Web-scale corpora often contain duplicated, noisy, or imbalanced content, making principled data selection essential for effective training. Recent work on data-constrained scaling laws further suggests that carefully selected high-quality data can outperform larger but lower-quality corpora, particularly when compute budgets are limited.

This project investigates intelligent data selection for mathematical language model training by comparing two complementary approaches:

1. **Hierarchical Clustering-Based Data Curation**
   Inspired by *Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach*, this method performs hierarchical k-means clustering on dataset embeddings to create balanced and diverse subsets of training data. The goal is to preserve broad topical and conceptual coverage while reducing redundancy and low-quality samples.

2. **RHO-1 Token-Level Selection**
   Inspired by *RHO-1: Not All Tokens Are What You Need*, this method operates at the token level rather than the document level. A reference model trained on curated mathematical data identifies informative tokens using per-token reference loss, allowing training to focus gradient updates only on tokens that provide meaningful learning signal.

Although both methods aim to improve training efficiency and model quality, they operate at different granularities: clustering selects *which documents* are used for training, while RHO-1 selects *which tokens* within those documents contribute to optimization. This project studies these approaches both independently and in combination to determine whether they provide complementary benefits.

Experiments are conducted using the `OpenWebMath` dataset and `TinyLlama-1.1B` as the base model. Evaluation is performed across a diverse suite of mathematical reasoning benchmarks including GSM8K, SVAMP, ASDiv, MAWPS, MATH, TAB MQA, SAT, and MMLU STEM. In addition to comparing clustering and RHO-1 individually, the project also explores:

* hybrid pipelines combining both methods,
* repeated training on smaller high-quality subsets,
* and comparisons between principled clustering-based selection and random subset sampling.

The goal of this work is to better understand how intelligent data selection can improve mathematical reasoning performance under constrained compute and data budgets.

---

# Method Comparison

| Aspect                        | Hierarchical Clustering                                | RHO-1                                      |
| ----------------------------- | ------------------------------------------------------ | ------------------------------------------ |
| Granularity                   | Dataset / document level                               | Token level                                |
| Main Idea                     | Select diverse and balanced documents                  | Select informative tokens                  |
| Inspired By                   | *Automatic Data Curation for Self-Supervised Learning* | *RHO-1: Not All Tokens Are What You Need*  |
| Selection Target              | Entire documents                                       | Individual tokens                          |
| Goal                          | Improve corpus quality and diversity                   | Improve gradient efficiency                |
| Core Mechanism                | Hierarchical k-means clustering on embeddings          | Reference-loss-based token filtering       |
| Potential Weakness            | May keep low-value tokens inside good documents        | Depends heavily on reference model quality |

---

# Experimental Pipelines

| Pipeline                     | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| Baseline CPT                 | Standard continual pretraining on OpenWebMath  |
| Clustering Only              | Train on clustering-selected subset            |
| RHO-1 Only                   | Token-level selective training                 |
| Hybrid (Clustering + RHO-1)  | Apply RHO-1 inside clustering-selected corpus  |
| Repeated High-Quality Subset | Train repeatedly on reduced curated subset     |
| Random Subset Baseline       | Randomly sampled subset with same token budget |

---

# Key Research Questions

| Question                                                                    | Motivation                                      |
| --------------------------------------------------------------------------- | ----------------------------------------------- |
| Does clustering improve mathematical reasoning compared to random sampling? | Measure value of structured data selection      |
| Does RHO-1 outperform dataset-level filtering alone?                        | Compare token-level vs document-level selection |
| Are clustering and RHO-1 complementary?                                     | Test hybrid pipeline effectiveness              |
| Can smaller repeated datasets outperform larger mixed-quality datasets?     | Validate data-constrained scaling laws          |

### Inspired by:
1. [Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach](https://arxiv.org/abs/2405.15613)
2. [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)
                                                                                  
&nbsp;

## Benchmark Comparison

The baseline model `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` was evaluated on the exact set of benchmark tasks used by the Rho-1 authors. The comparison results are shown in the table below.
## Few-shot CoT reasoning results 

| Evaluation                   | GSM8K | SVAMP | ASDiv | MAWPS | TAB | MQA | MMLU STEM | SAT‡ | MATH |
|------------------------------|-------|-------|-------|-------|-----|-----|-----------|------|------|
| TinyLlama-1.1B (Rho-1 paper) | 2.9 | 11.0 | 18.1 | 20.4 | 12.5 | 14.6 | 16.1 | 21.9 | 3.2 |
| TinyLlama-1.1B &nbsp;(My evaluation) | 2.7 | 10.9 | 17.9 | 20.5 | 12.5 | 13.9 | 16.4 | 21.9 | 2.2 |

                                                                                                    
&nbsp;
                                                                                                 
## Continual Pretraining (CPT) Results

Performed continual pretraining (CPT) of TinyLlama-1.1B on the `OpenWebMath` dataset following the RHO-1 study. The table below compares our CPT results with the results reported by the RHO-1 authors.
## Few-shot CoT reasoning results 

| Evaluation                 | GSM8K | SVAMP | ASDiv | MAWPS | TAB | MQA | MMLU STEM | SAT‡ | MATH |
|----------------------------|-------|-------|-------|-------|-----|-----|-----------|------|------|
| TinyLlama-1.1B CT (Rho-1 paper) | 6.4 | 21.7 | 36.7 | 47.7 | 17.9 | 13.9 | 23.0 | 25.0 | 2.4 |
| TinyLlama-1.1B CT (My evaluation) | 6.0 | 19.3 | 31.2 | 42.1 | 14.4 | 12.4 | 21.0 | 28.1 | 3.9 |


<table>
<thead>
<tr>
<th>Model</th>
<th>|θ|</th>
<th>Data</th>
<th>Uniq. Toks</th>
<th>For.</th>
<th>Back.</th>
<th>GSM8K</th>
<th>SVAMP</th>
<th>ASDiv</th>
<th>MAWPS</th>
<th>TAB</th>
<th>MQA</th>
<th>MMLU STEM</th>
<th>SAT</th>
<th>MATH</th>
<th>Avg.</th>
</tr>
</thead>

<tbody>

<tr style="background-color:#f2f2f2;">
<td><b>TinyLlama_CT</b></td>
<td>1.1B</td>
<td>OWM</td>
<td>14B</td>
<td>15B</td>
<td>15B</td>
<td>6.4</td>
<td>21.7</td>
<td>36.7</td>
<td>47.7</td>
<td>17.9</td>
<td>13.9</td>
<td>23.0</td>
<td>25.0</td>
<td>2.4</td>
<td>21.6</td>
</tr>

<tr style="background-color:#ffffff;">
<td><b>TinyLlama_CT</b></td>
<td>1.1B</td>
<td>OWM</td>
<td>14B</td>
<td>15B</td>
<td>15B</td>
<td>5.3</td>
<td>19.9</td>
<td>32.7</td>
<td>42.1</td>
<td>14.7</td>
<td>11.9</td>
<td>20.4</td>
<td>25.0</td>
<td>3.5</td>
<td>19.5</td>
</tr>

<tr style="background-color:#e6f7ff;">
<td><b>Rho-1-Math (SLM)</b></td>
<td>1.1B</td>
<td>OWM</td>
<td>14B</td>
<td>15B</td>
<td>9B</td>
<td><b>29.8</b></td>
<td><b>49.2</b></td>
<td><b>61.4</b></td>
<td><b>79.8</b></td>
<td><b>25.8</b></td>
<td><b>30.4</b></td>
<td><b>24.7</b></td>
<td><b>28.1</b></td>
<td><b>14.0</b></td>
<td><b>38.1</b></td>
</tr>

</tbody>
</table>
&nbsp;

> **Note:** The TinyLlama-CT model was not publicly released, comparisons are based on reported metrics.

&nbsp;

## Hierarchical Clustering on Open Web Math Dataset Embeddings
Embedding Model: `intfloat/e5-base-v2`
