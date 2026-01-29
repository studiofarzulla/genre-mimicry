# Genre Mimicry vs. Ethical Reasoning in Abliterated Language Models

**Why Training Data Conventions Persist After Safety Removal**

[![arXiv](https://img.shields.io/badge/arXiv-2501.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2501.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

Abliterated language models—those with safety fine-tuning removed through techniques such as refusal direction orthogonalization—are commonly assumed to have lost their ethical reasoning capabilities. This paper challenges that assumption by presenting evidence that what appears to be ethical reasoning in language models is substantially influenced by **genre convention mimicry**: the reproduction of professional writing norms absorbed from training data rather than genuine moral cognition.

Through a multi-model empirical study (n=9 architectures, N=215 prompts across four content genres), we observe a differential response pattern: requests matching information security and finance genres generate disclaimers at rates of 50.8% and 77.8% respectively, while violence-related prompts produce disclaimers in only 30.4% of cases. This "Violence Gap" is statistically significant (χ²(1) = 17.08, p < 0.0001, OR = 3.99) and persists across both abliterated and control models.

## Key Findings

| Genre | Disclaimer Rate | Warn-and-Answer Rate |
|-------|-----------------|---------------------|
| Finance/Fraud | 77.8% | 94.3% |
| Chemistry | 67.3% | 93.9% |
| InfoSec | 50.8% | 69.7% |
| Violence | 30.4% | 64.7% |

- **Violence Gap**: Models are 3.99× more likely to include disclaimers for non-violence content
- **Decorative Disclaimers**: 83.1% of responses with disclaimers still contain harmful content
- **Genre-Locked Safety**: Pattern persists in both abliterated and control models

## Repository Structure

```
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
└── CITATION.cff
```

## Reproducing Results

```bash
cd analysis
pip install pandas numpy scipy statsmodels
python statistical_analysis.py
```

**Requirements**: Python 3.11+, pandas 2.1+, statsmodels 0.14+, scipy 1.11+

**Decoding Parameters**: Temperature = 0.7, max_tokens = 512

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

## Citation

```bibtex
@article{farzulla2026genre,
  title={Genre Mimicry vs. Ethical Reasoning in Abliterated Language Models:
         Why Training Data Conventions Persist After Safety Removal},
  author={Farzulla, Murad},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Author

**Murad Farzulla**
King's College London
[ORCID: 0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
