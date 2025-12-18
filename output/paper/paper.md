# Gender Bias in LLaMA-3 Embeddings: Implications for LinkedIn-Style Retrieval Systems

**Christopher S. Penn**
Chief Data Scientist, TrustInsights.ai

**Katie Robbert**
Chief Executive Officer, TrustInsights.ai

---

## Abstract

Large language models increasingly power professional network retrieval systems, with LinkedIn recently deploying LLaMA-3 for member and content embeddings. We investigate whether these embeddings exhibit gender bias by measuring semantic drift when we attribute identical professional content to names of different perceived genders. We constructed 406 paired LinkedIn-style posts with identical text content, headlines, and professional context, differing only in author name (male versus female variants), and extracted embeddings using LLaMA-3.2-3B's hidden states with mean pooling—replicating LinkedIn's published methodology. We find systematic bias: mean cosine similarity between paired embeddings is 0.994 (not the expected 1.0), with Cohen's d = -0.93 (large effect) and p < 0.0001 across both parametric and non-parametric statistical tests. This approximately 0.6% embedding deviation, while small per-pair, represents systematic differential treatment that compounds across retrieval, ranking, and recommendation systems—potentially affecting search visibility for millions of professionals. We release our dataset, code, and statistical framework to enable reproducible bias auditing of LLM-based retrieval systems.

**Code & Data:** https://github.com/trustinsights/linkedinbias

---

## 1. Introduction

LinkedIn serves over one billion professionals worldwide, functioning as critical infrastructure for job seeking, professional networking, and career advancement. Recent research reveals that LinkedIn has deployed LLaMA-3, Meta's large language model, to power their retrieval system for member and content search (arXiv:2510.14223). Critically, author name serves as an explicit feature in embedding construction, raising questions about whether the model treats identical content differently based on the perceived gender of the author's name.

If embeddings differ for identical professional content based solely on name, the retrieval system exhibits bias at its foundational layer. This bias propagates through search results, connection recommendations, and job matching algorithms. Unlike downstream ranking systems that can potentially correct for bias using behavioral signals, embedding-stage bias affects all users at the first filter stage. Cold-start users—new professionals without established engagement histories—face particular vulnerability, as LinkedIn's own research acknowledges significant performance gaps for this population (arXiv:2501.16450v4).

Our research addresses a fundamental question:

> *Does LLaMA-3 produce systematically different embeddings for identical professional content when only the author's perceived gender differs?*

To answer this question, we constructed 406 paired posts containing identical professional content, differing only in author name (male versus female phonetic variants). This sample size provides exceptional statistical power, exceeding the theoretical minimum required to detect this effect size by over 30 times. We extracted embeddings using LinkedIn's published methodology—hidden state extraction with mean pooling—and applied rigorous statistical analysis with multiple robustness checks. We found large, systematic bias: Cohen's d = -0.93, indicating that the model consistently encodes gendered names differently even when all other content remains identical.

### Contributions

1. We present the **first public audit** of gender bias in LinkedIn's LLaMA-3-based retrieval embeddings, replicating their published methodology.
2. We apply the **established paired-content methodology** from fairness research (Bertrand & Mullainathan, 2004; Wilson & Caliskan, 2024) to professional network profile embeddings.
3. We release an **open-source auditing tool and dataset** enabling community verification and extension of our findings.
4. We provide a **statistical framework** combining parametric and non-parametric validation suitable for embedding bias research.

---

## 2. Related Work

### 2.1 Bias in Word and Sentence Embeddings

The Word Embedding Association Test (WEAT) demonstrated that word embeddings trained on large corpora contain human-like biases, associating male names with career terms and female names with family terms (Caliskan et al., 2017). Subsequent work extended these findings to sentence-level encoders, showing that contextualized representations from BERT and similar models exhibit gender bias in downstream tasks (May et al., 2019; Kurita et al., 2019).

The foundational work of Bertrand & Mullainathan (2004) on name-based discrimination in hiring—finding that resumes with stereotypically white names received 50% more callbacks than identical resumes with stereotypically Black names—established the real-world impact of name-based differential treatment.

Most relevant to our work, Wilson & Caliskan (2024) audited gender and racial bias in resume screening using language model retrieval. Their study used paired resumes with 120 demographically-associated names to measure bias in Massive Text Embedding models, finding significant disparities in retrieval ranking. Our work extends this methodology to professional network search, specifically targeting LinkedIn's LLaMA-3-based architecture.

### 2.2 LinkedIn's Retrieval Architecture

LinkedIn recently published details of their LLM-based retrieval system (arXiv:2510.14223). Their architecture fine-tunes LLaMA-3 as a dual encoder to generate dense embeddings for both members (queries) and content (items), enabling semantic search across their billion-user platform. In this dual encoder setup, a single shared LLM processes member and item prompts separately, producing embeddings that are compared via cosine similarity. The system constructs embeddings using member features including name, headline, and content text, applying mean pooling over hidden states from the model's final transformer layer.

Their published results show that the base LLaMA-3 model achieves Recall@10 of 0.24, while fine-tuning on LinkedIn-specific data improves this to 0.42—a 74% relative improvement. Additional LinkedIn engineering publications describe their feed infrastructure, including FishDB for generic retrieval, speculative decoding optimizations for their hiring assistant, and recent work on accelerating recommendation systems using SGLang.

### 2.3 Gap in the Literature

While Wilson & Caliskan (2024) demonstrated bias in retrieval embeddings for resume screening, their work used general-purpose embedding models rather than platform-specific systems. No public audit has examined gender bias specifically in LinkedIn's LLaMA-3-based retrieval architecture, where author name serves as an explicit embedding feature.

---

## 3. Methodology

### 3.1 Experimental Design

We employ a within-subject paired comparison design to isolate the effect of gendered names on embedding representations:

- **Independent variable:** Author name (male versus female phonetic variant)
- **Dependent variable:** Cosine similarity between embedding vectors for paired posts
- **Null hypothesis:** H₀: μ_similarity = 1.0 (no difference)

Because each pair contains identical text content, any difference in embeddings can only arise from the author name.

### 3.2 Data Collection

We collected public LinkedIn posts using screen recording on an iPhone 15 running the LinkedIn mobile application. The researcher scrolled through their public feed twice, capturing visible posts. Each session lasted 5 minutes and captured approximately 75 posts. To mitigate potential bias from the researcher's AI-focused feed, we supplemented with hashtag-based searches across diverse professional domains, gathering an additional 250 posts:

- Healthcare: #healthcare, #nursing
- Social issues: #black, #poverty
- Creative: #music
- Education: #teachers
- Professional: #leadership

We processed screen recordings using Google Gemini 3 Pro in Google AI Studio for video-to-JSON transcription. The model extracted three fields from each visible post: author name, professional headline (as displayed, which may be truncated), and post text content. The model excluded sponsored content and advertisements during transcription. See Appendix A for the complete transcription prompt.

We used Google Gemini 3 Pro to generate gender-coded name variants from the transcribed data. For each post, we produced both a male-coded and female-coded version by prompting the model to transform names while preserving ethnic and cultural consistency. For example, "Christina Applegate" becomes "Christian Applegate," and "Robert Miller" becomes "Roberta Miller." Critically, Gemini automatically maintained ethnic consistency during transformation—names from non-Western naming conventions received culturally appropriate gender transformations, not Western substitutions.

The transformation process used separate prompts for male and female dataset generation (see Appendix A for complete prompts). We stripped emoji characters from name fields during processing. All other fields—professional headline and post text content—remained identical between pairs. The final dataset contains 406 unique content pairs.

| Attribute | Value |
|-----------|-------|
| Total pairs | 406 |
| Collection period | December 2025 |
| Sources | Feed + hashtag search |
| Professional domains | 5+ |

### 3.3 Embedding Generation

We replicate LinkedIn's published dual encoder methodology, applying the item-side (content) encoding process:

1. Use LLaMA-3.2-3B (bfloat16 precision)
2. Extract hidden states from the final transformer layer
3. Apply mean pooling across all tokens
4. Obtain 3072-dimensional embedding vector

**Prompt format:**
```
Author Name: {name}
Author Headline: {headline}
Post Text: {text_content}
```

### 3.4 Statistical Analysis

- **Primary test:** One-sample t-test against μ = 1.0
- **Robustness checks:** Wilcoxon signed-rank test, permutation test (10,000 resamples), bootstrap CI (BCa method)
- **Effect size:** Cohen's d with interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, ≥0.8 large

---

## 4. Results

### 4.1 Primary Findings

| Metric | Value |
|--------|-------|
| N (pairs) | 406 |
| Mean similarity | 0.9940 |
| Standard deviation | 0.0064 |
| 95% CI | [0.9934, 0.9946] |
| Expected (no bias) | 1.0000 |
| Deviation | -0.0060 |

### 4.2 Statistical Tests

| Test | Statistic | p-value |
|------|-----------|---------|
| One-sample t-test | t = -18.9 | < 0.0001 |
| Wilcoxon signed-rank | W = 0 | < 0.0001 |
| Permutation test | — | < 0.0001 |

All tests reject H₀: μ = 1.0.

### 4.3 Effect Size

**Cohen's d = -0.927** (Large effect)

95% CI: [-1.02, -0.83]

### 4.4 Robustness Analysis

Effect size stability as sample grew:

| N | Mean Sim. | Std Dev | Cohen's d |
|---|-----------|---------|-----------|
| 76 | 0.9933 | 0.0083 | -0.808 |
| 158 | 0.9934 | 0.0074 | -0.894 |
| 210 | 0.9939 | 0.0067 | -0.905 |
| 285 | 0.9937 | 0.0065 | -0.960 |
| 334 | 0.9939 | 0.0068 | -0.902 |
| 406 | 0.9940 | 0.0064 | -0.927 |

Key observations:
- Mean similarity stable at ~0.994 across all sample sizes
- Effect size stabilized in "large" range rather than regressing toward zero
- 5× sample increase did not diminish the effect

---

## 5. Discussion

### 5.1 Interpretation

LLaMA-3 produces systematically different embeddings for identical professional content when author names differ by perceived gender. The ~0.6% deviation represents a **large effect** (d = -0.93) because it occurs consistently across all 406 tested pairs.

The model encodes gender-associated information from names into the embedding representation, even when actual content remains identical. This is not a function of content quality—the embedding space itself treats gendered names differently.

### 5.2 Implications

At LinkedIn's scale of over one billion users:
- Small systematic biases aggregate to substantial effects
- Affected individuals may never appear in search results—not because of qualifications, but because of how the model encodes their name
- Cold-start users (new professionals) cannot benefit from downstream correction systems

### 5.3 Limitations

1. **Base model vs. fine-tuned:** We tested base LLaMA-3; LinkedIn uses a fine-tuned variant. Fine-tuning could reduce or amplify bias.

2. **Name selection:** Limited to Western gendered name pairs. May not generalize to other naming conventions.

3. **Content domain:** LinkedIn-style professional posts only.

4. **No downstream measurement:** We measured embedding similarity, not actual search rankings or user outcomes.

5. **Association not causation:** We demonstrate bias exists; we cannot prove it causes harm in production.

6. **Single model family:** Only tested LLaMA-3.

---

## 6. Conclusion

We present the first public audit of gender bias in LLM embeddings for professional network retrieval. Using LinkedIn's published methodology, we generated embeddings for 406 paired posts—identical content with only author name varying by perceived gender—and found systematic bias:

- **Mean cosine similarity:** 0.994 (vs. expected 1.0)
- **Cohen's d:** -0.93 (large effect)
- **p-value:** < 0.0001

This finding held robustly across 5× sample growth and multiple statistical tests. The bias manifested consistently across diverse professional domains.

### Future Work

- Test fine-tuned models
- Extend to other demographic attributes
- Measure downstream effects on search rankings
- Cross-cultural name studies
- Develop bias mitigation techniques

---

## Ethics Statement

This research used publicly available LinkedIn posts collected via screen recording of the LinkedIn mobile application. Our released dataset contains author first and last names, professional headlines, and post text content. No other identifying information is present—we did not collect profile URLs, usernames, employment history, or contact information.

Our dataset contains a mixture of original and transformed names. When an original author's name already matched the target gender coding (e.g., a male-coded name for the male dataset), we retained that name as-is. The LLM modified only names requiring gender transformation. Consequently, the male-coded dataset contains real names for originally male-coded authors, while the female-coded dataset contains real names for originally female-coded authors. The gender-flipped counterparts in each dataset are synthetic transformations.

We collected all posts from public LinkedIn feeds. We release our methodology and tools to enable responsible bias auditing, with the goal of improving fairness in AI-powered professional platforms. We recognize that bias auditing research can itself be misused; we encourage use of these methods for improving system fairness rather than exploiting identified biases.

---

## Data & Code Availability

All materials publicly available at:
**https://github.com/trustinsights/linkedinbias**

- Code: `src/` directory
- Data: `data/` directory
- Results: `output/` directory
- License: MIT (code), CC BY 4.0 (data)

---

## References

1. Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183-186.

2. Gupta, R. et al. (2024). Large Scale Retrieval for the LinkedIn Feed using Causal Language Models. *arXiv:2510.14223*.

3. LinkedIn AI Team. (2025). 360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation. *arXiv:2501.16450v4*.

4. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

5. May, C. et al. (2019). On Measuring Social Biases in Sentence Encoders. *NAACL-HLT 2019*.

6. Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? *American Economic Review*, 94(4), 991-1013.

7. LinkedIn Engineering. (2024). FishDB: A generic retrieval engine for scaling LinkedIn's feed. *LinkedIn Engineering Blog*.

8. LinkedIn Engineering. (2024). Accelerating LLM inference with speculative decoding: Lessons from LinkedIn's Hiring Assistant. *LinkedIn Engineering Blog*.

9. Kurita, K. et al. (2019). Measuring Bias in Contextualized Word Representations. *Workshop on Gender Bias in NLP*.

10. Meta AI. (2024). Llama 3.2: Lightweight, open models for on-device AI.

11. Shimizu, S. et al. (2025). Turbocharging LinkedIn's Recommendation Systems with SGLang. *LinkedIn Engineering Blog*.

---

## Appendix A: Data Collection Prompts

We used Google Gemini 3 Pro in Google AI Studio for both transcription and name transformation. We used the following prompts in our data collection pipeline.

### A.1 Video Transcription Prompt

> You are a transcription expert skilled in transcribing social media content. Your remit today is to transcribe the LinkedIn posts shown in the associated video attached. You will transcribe three fields. First is the name of the poster. Second is the headline of the poster. Note that the headline may be truncated due to display constraints. Transcribe exactly what you see. Third is the text content of their post. Exclude sponsored content and advertisements. For posts which contain multimedia such as images or video, transcribe the text to the post exactly and then add a placeholder for audio or video indicating where the multimedia content is. You will return your content in JSON format with three fields name, headline, and text content.

### A.2 Male Dataset Generation Prompt

> Our next task is to produce 2 sets of data based on this JSON data. Here's what we're going to do. For each name field in the JSON object, we are going to produce a male name dataset first, then a female name dataset. You'll substitute the nearest male-coded name in each field if the existing name is ambiguous or female coded.
>
> Example: "name": "Christina Applegate" becomes "name": "Christian Applegate"
>
> Produce the male dataset first. If a name is already male-coded, leave it as is. The new dataset should be in the same JSON format and have identical headline and text_content. The only change we are making is the name field.
>
> Critically important: strip away emoji from the name field in the results.
>
> Produce ONLY the male dataset now.

### A.3 Female Dataset Generation Prompt

> Now perform the same task, but making each name female-coded. Keep all other data the same, as you did in the male data set.

---

*© 2025 TrustInsights.ai. Released under CC BY 4.0.*
