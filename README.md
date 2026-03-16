# Capstone: Smart data selection for language modeling

Training TinyLlama-1.1B on the same set as Rho-1 but with a new type of filtering inspired by the works below. Using hierarchical clustering on token representations that are computed using the reference model (keeping the contexts).

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

&nbsp;

> **Note:** The TinyLlama-CT model was not publicly released, comparisons are based on reported metrics.

&nbsp;

## Hierarchical Clustering on Open Web Math Dataset Embeddings
Embedding Model: `intfloat/e5-base-v2`
