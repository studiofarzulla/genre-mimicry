# Genre Mimicry vs. Ethical Reasoning in Abliterated Language Models

**Why Training Data Conventions Persist After Safety Removal**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17957694-blue.svg)](https://doi.org/10.5281/zenodo.17957694)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-Preprint-green.svg)](https://doi.org/10.5281/zenodo.17957694)

**Discussion Paper DP-2503** | [Dissensus AI](https://dissensus.ai)

## Abstract

Abliterated language models---those with safety fine-tuning removed through techniques such as refusal direction orthogonalization---are commonly assumed to have lost their ethical reasoning capabilities. This paper challenges that assumption by presenting evidence that what appears to be ethical reasoning in language models is substantially influenced by genre convention mimicry: the reproduction of professional writing norms absorbed from training data rather than genuine moral cognition. Through a multi-model empirical study (n=9 architectures, N=215 prompts across four content genres), we observe a differential response pattern that warrants further safety research. Requests matching information security and finance genres generate disclaimers at rates of 50.8% and 77.8% respectively, while violence-related prompts produce disclaimers in only 30.4% of cases. This "Violence Gap" is statistically significant (chi-squared(1) = 17.08, p < 0.0001, OR = 3.99) and persists across both abliterated and control models. GEE logistic regression with cluster-robust standard errors confirms Finance/Fraud (OR = 9.63, p < 0.001) and Chemistry (OR = 5.21, p = 0.034) effects. We introduce the concept of Genre Vulnerability---content domains exhibiting reduced safety behaviors due to the absence of native safety conventions in training corpora---and extend our analysis to a theoretical framework (the "Parity Thesis") proposing that human reasoning is similarly constrained by training distributions.

## Key Findings

| Finding | Result |
|---------|--------|
| Violence Gap | Models 3.99x more likely to include disclaimers for non-violence content (p < 0.0001) |
| Finance/Fraud disclaimer rate | 77.8% -- highest across all genres |
| Violence disclaimer rate | 30.4% -- lowest across all genres |
| Decorative disclaimers | 83.1% of responses with disclaimers still contain harmful content |
| Genre persistence after abliteration | Violence Gap persists in both abliterated and control models |

## Models Evaluated

| Model | Type | Parameters |
|-------|------|------------|
| Gemma3-27B-Abl | Abliterated | 27B |
| Qwen2.5-32B-Abl | Abliterated | 32B |
| Qwen2.5-32B-Abl-2 | Abliterated | 32B |
| Qwen3-8B-Abl | Abliterated | 8B |
| Qwen3-VL-8B-Abl | Abliterated | 8B |
| Llama-MoE-18B-Abl | Abliterated | 18.4B |
| GPT-OSS-20B-Abl | Abliterated | 20B |
| Qwen3-30B | Control | 30B |
| Devstral-Small | Control | 2.5B |

## Keywords

AI safety, abliteration, language models, genre theory, training data, alignment, professional norms

## Repository Structure

```
genre-mimicry/
├── paper/
│   ├── genre-mimicry-arxiv.tex    # LaTeX source
│   └── genre-mimicry-arxiv.pdf    # Compiled paper
├── data/
│   └── genre_mimicry_results_*.jsonl  # Raw response data (9 models)
├── analysis/
│   ├── statistical_analysis.py    # Main analysis script
│   ├── analysis_results.json      # Computed statistics
│   ├── harm_scores_ollama.jsonl   # Llama Guard classifications
│   ├── summary_by_model_genre.csv # Summary statistics
│   └── results_tables.tex         # LaTeX tables
├── CITATION.cff
└── LICENSE
```

## Reproducing Results

```bash
cd analysis
pip install pandas numpy scipy statsmodels
python statistical_analysis.py
```

**Requirements**: Python 3.11+, pandas 2.1+, statsmodels 0.14+, scipy 1.11+

## Citation

```bibtex
@article{farzulla2026genre,
  author  = {Farzulla, Murad},
  title   = {Genre Mimicry vs. Ethical Reasoning in Abliterated Language Models: Why Training Data Conventions Persist After Safety Removal},
  year    = {2026},
  journal = {Dissensus AI Discussion Paper DP-2503},
  doi     = {10.5281/zenodo.17957694}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
