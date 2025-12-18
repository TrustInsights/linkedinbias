# Gender Bias in LLaMA-3 Embeddings: Paper Materials

## Authors
- **Christopher S. Penn**, Chief Data Scientist, TrustInsights.ai
- **Katie Robbert**, Chief Executive Officer, TrustInsights.ai

## Files

| File | Description |
|------|-------------|
| `main.tex` | LaTeX source (ACL 2024 format) |
| `references.bib` | BibTeX bibliography |
| `paper.md` | Markdown version (for accessibility) |
| `README.md` | This file |

## Compiling the LaTeX

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- ACL style files (download from [ACL](https://github.com/acl-org/acl-style-files))

### Compile Commands

```bash
# Using pdflatex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Or using latexmk (recommended)
latexmk -pdf main.tex
```

### ACL Style Files

Download the ACL 2024 style files and place in this directory:
- `acl.sty`
- `acl_natbib.bst`

From: https://github.com/acl-org/acl-style-files

## arXiv Submission

For arXiv submission:
1. Compile to PDF
2. Package source files: `main.tex`, `references.bib`, style files
3. Submit to cs.CL (Computation and Language) category
4. Recommended license: CC BY 4.0

## Code and Data

All code and data referenced in this paper available at:
https://github.com/trustinsights/linkedinbias

## Citation

```bibtex
@article{penn2025genderbias,
    title={Gender Bias in LLaMA-3 Embeddings: Implications for LinkedIn-Style Retrieval Systems},
    author={Penn, Christopher S. and Robbert, Katie},
    journal={arXiv preprint},
    year={2025}
}
```
