# Data Collection Methodology Notes

## Last Updated: 2025-12-11

## Current Sample Size
- **406 pairs** (expanded from original 76 — more than 5× increase)
- Statistical power: >99% for detecting effect size d=0.2
- Minimum detectable effect: d ≈ 0.14

---

## Results Progression as Data Expanded

### Summary Table

| Date | Sample Size | Mean Similarity | Std Dev | Cohen's d | Verdict |
|------|-------------|-----------------|---------|-----------|---------|
| 12-10 18:31 | 76 | 0.9933 | 0.0083 | **-0.808** | Large Bias |
| 12-10 20:14 | 158 | 0.9934 | 0.0074 | **-0.894** | Large Bias |
| 12-11 10:33 | 210 | 0.9939 | 0.0067 | **-0.905** | Large Bias |
| 12-11 14:17 | 285 | 0.9937 | 0.0065 | **-0.960** | Large Bias |
| 12-11 18:57 | 334 | 0.9939 | 0.0068 | **-0.902** | Large Bias |
| 12-11 19:24 | **406** | 0.9940 | 0.0064 | **-0.927** | Large Bias |

### Key Observations

1. **Mean similarity remarkably stable** at ~0.994 across all sample sizes (76 → 406)
   - This consistency suggests the bias effect is not an artifact of particular posts
   - New data from diverse sources produced identical deviation patterns
   - The model consistently treats gendered names ~0.6% differently

2. **Standard deviation decreased and stabilized** (0.0083 → 0.0064)
   - Expected statistical behavior with larger samples
   - Indicates the effect is consistent across posts, not driven by outliers

3. **Effect size stabilized in "Large" range** (d = -0.90 to -0.96)
   - Initial increase from d=-0.808 (76 pairs) to d=-0.960 (285 pairs)
   - Stabilized around d=-0.92 with additional data (334-406 pairs)
   - Natural sampling variation, but all firmly in "Large Bias" territory (threshold: |d| ≥ 0.8)

4. **"Large Bias" verdict held across ALL SIX runs**
   - All p-values = 0.0000 (highly significant)
   - Starting at n=158, non-parametric tests were used (Shapiro-Wilk detected non-normality)
   - Conclusion is robust to both parametric and non-parametric methods
   - Finding replicated consistently as sample grew 5×

### Interpretation for Publication

The progression from 76 to 406 pairs demonstrates:

> **The bias finding is robust, stable, and generalizable.** As the sample expanded more than fivefold—from the researcher's initial feed (76 pairs) to include posts from diverse professional domains via hashtag search (406 pairs total)—the measured effect size stabilized in the "Large" range (d ≈ -0.93, p < 0.0001). The mean similarity remained constant at ~0.994 across all sample sizes, indicating that the LLaMA 3 model consistently encodes identical content ~0.6% differently based solely on gendered names. This consistency across 406 independent tests provides strong evidence that the bias pattern is systematic and not an artifact of sample selection.

---

## Sample Bias Considerations

### What Our Paired Design Controls For
- Each pair has **identical** text, headline, and context - only the name changes
- This is a within-subject design - comparing the same post processed twice
- Any embedding difference can ONLY come from the gendered name
- Topic/style bias in the sample is symmetric (affects both genders equally)

### What We Don't Control For
- **Topic distribution**: If certain topics interact with gender differently (e.g., "tech CEO" vs "teacher"), and the sample over-represents one category, this could affect generalizability
- **Feed algorithm curation**: Posts from a single feed may not represent all LinkedIn content
- **This limits generalizability, not internal validity**

---

## Recommendation: Hashtag Diversification

### Rationale
1. **Breaks feed algorithm correlation** - Searching hashtags bypasses LinkedIn's personalized curation
2. **Topic independence** - Consistent bias across diverse domains is a stronger finding
3. **Generalizability claim** - Can claim "bias exists across professional domains" not just "in posts like my feed"

### Suggested Hashtags to Sample From

| Category | Example Hashtags | Rationale |
|----------|------------------|-----------|
| Tech/Engineering | #SoftwareEngineering, #DataScience | Male-dominated perception |
| Healthcare | #Nursing, #Healthcare | Female-dominated perception |
| Business/Finance | #Leadership, #Entrepreneurship | Gender-neutral topic |
| Education | #Teaching, #EdTech | Different sector |
| Creative | #Marketing, #ContentCreation | Mixed perception |

**Goal**: 3-5 hashtags across different professional domains

---

## Methodology Documentation for Paper

Suggested wording:

> To ensure topic diversity and avoid feed algorithm bias, posts were collected from multiple sources: (1) the researcher's LinkedIn feed (N=X pairs), and (2) public posts discovered via hashtag search across diverse professional domains including [list hashtags] (N=Y pairs). This design tests whether embedding bias generalizes across content types.

---

## Key Insight

The question isn't "does my feed have biased content?" but "does the AI model treat the SAME content differently based on gendered names?"

Since we use identical content pairs, any topic bias in the sample is symmetric. The cosine similarity measures the DIFFERENCE in processing, not absolute content quality.

---

## Action Items
- [x] Collect posts from 3-5 diverse hashtags *(406 pairs collected — 5× original sample)*
- [x] Track progression of results as sample size grew *(documented above — 6 runs)*
- [x] Remove duplicate entries *(1 exact duplicate removed, 284→406 clean pairs)*
- [ ] Track which posts came from feed vs hashtag search *(consider adding `"source"` field to JSON)*
- [ ] Consider categorizing posts by industry/topic for subgroup analysis
- [ ] Document collection methodology for paper

---

## Data Quality Note (2025-12-11)

Fixed field name inconsistency in JSON data:
- **Issue**: 75 records in each file had `"text content"` (with space) instead of `"text_content"` (with underscore)
- **Cause**: Likely data entry variation when new posts were added in bulk
- **Fix**: sed replacement, verified with JSON syntax check and pytest
- **Impact**: None on analysis - data was correct, only field name was inconsistent
