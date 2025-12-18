# Gender Bias in LLaMA-3 Embeddings

**Auditing bias in LinkedIn-style retrieval systems using LLM embeddings**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)

---

## Key Finding

**LLaMA-3 produces systematically different embeddings for identical professional content when author names differ by perceived gender.**

| Metric | Value |
|--------|-------|
| Mean cosine similarity | 0.994 (expected: 1.0) |
| Cohen's d | **-0.93** (large effect) |
| p-value | < 0.0001 |
| Sample size | 406 paired posts |

This ~0.6% embedding deviation represents systematic differential treatment that compounds across retrieval, ranking, and recommendation systems—potentially affecting search visibility for millions of professionals.

---

## Overview

LinkedIn recently deployed LLaMA-3 to power their retrieval system for member and content search ([arXiv:2510.14223](https://arxiv.org/abs/2510.14223)). Author name serves as an explicit feature in embedding construction.

We investigated whether these embeddings exhibit gender bias by:

1. Constructing 406 paired LinkedIn-style posts with identical content
2. Varying only the author name (male vs. female variants)
3. Extracting embeddings using LinkedIn's published methodology (hidden state extraction with mean pooling)
4. Applying rigorous statistical analysis

**Result:** Large, systematic bias confirmed across all tests.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/TrustInsights/linkedinbias.git
cd linkedinbias

# Install dependencies
poetry install
```

### Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running locally with LLaMA-3.2-3B
- ~8GB RAM for embedding generation

---

## Usage

### Run the Audit

```bash
# Start LM Studio with LLaMA-3.2-3B model first
poetry run python -m src.main audit
```

### Configuration

Edit `config.yml` to customize:

```yaml
lm_studio:
  base_url: "http://127.0.0.1:1234/v1"
  model: "llama-3.2-3b"

paths:
  male_data: "data/maledata.json"
  female_data: "data/femaledata.json"
  output_db: "output/audit.db"
```

---

## Data

### Dataset Structure

| File | Description |
|------|-------------|
| `data/maledata.json` | 406 posts with male-coded author names |
| `data/femaledata.json` | 406 posts with female-coded author names |

Each record contains:
- `name`: Author name (gender-transformed)
- `headline`: Professional headline (as displayed on LinkedIn)
- `text_content`: Post text (identical between pairs)

### Collection Methodology

- Source: Public LinkedIn feed + hashtag searches (#healthcare, #nursing, #leadership, #teachers, #music, #black, #poverty)
- Collection: Screen recording on iPhone 15, December 2025
- Transcription: Google Gemini 3 Pro
- Name transformation: Culturally-consistent gender variants (e.g., "Christina" → "Christian")

---

## Results

### Statistical Tests

| Test | Statistic | p-value |
|------|-----------|---------|
| One-sample t-test | t = -18.9 | < 0.0001 |
| Wilcoxon signed-rank | W = 0 | < 0.0001 |
| Permutation test (10,000) | — | < 0.0001 |

### Effect Size Stability

The effect remained robust across 5× sample growth:

| N | Mean Similarity | Cohen's d |
|---|-----------------|-----------|
| 76 | 0.9933 | -0.808 |
| 158 | 0.9934 | -0.894 |
| 406 | 0.9940 | **-0.927** |

---

## Repository Structure

```
linkedinbias/
├── src/                    # Source code
│   ├── main.py            # CLI entry point
│   ├── analysis.py        # Embedding comparison logic
│   ├── client.py          # LM Studio API client
│   ├── database.py        # SQLAlchemy persistence
│   └── stats.py           # Statistical analysis
├── data/                   # Paired datasets
├── output/
│   ├── paper/             # Research paper (LaTeX + PDF)
│   ├── analysis/          # Methodology documentation
│   └── audit.db           # Results database
├── config.yml             # Configuration
└── pyproject.toml         # Dependencies
```

---

## Citation

```bibtex
@article{penn2025genderbias,
    title={Gender Bias in LLaMA-3 Embeddings: Implications for LinkedIn-Style Retrieval Systems},
    author={Penn, Christopher S. and Robbert, Katie},
    year={2025},
    url={https://github.com/TrustInsights/linkedinbias}
}
```

---

## Paper

The full research paper is available in [`output/paper/`](output/paper/):

- [PDF](output/paper/gender-bias-llama-3-embeddings-2025-12-18-trust-insights.pdf)
- [LaTeX source](output/paper/main.tex)
- [Markdown version](output/paper/paper.md)

---

## Authors

- **Christopher S. Penn** — Chief Data Scientist, [TrustInsights.ai](https://trustinsights.ai)
- **Katie Robbert** — Chief Executive Officer, [TrustInsights.ai](https://trustinsights.ai)

---

## License

- **Code:** MIT License
- **Data & Paper:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Acknowledgments

This research used the [Hugging Face Transformers](https://huggingface.co/transformers/) library and [LM Studio](https://lmstudio.ai/) for local inference.
