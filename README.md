# Exploring Translation Complexity: From Sentence Readability to Cognitive Load and LLM Translation Quality

---

## 1. What the code does

The notebook implements the four phases described in the report:

| Phase | Goal | Notebook Section |
|---|---|---|
| 1 | Train a sentence-complexity model on the CLEAR corpus and inspect feature weights | Sections 1–4 |
| 2 | Relate sentence features to eye-tracking / keystroke measures from CRITT TPR-DB NJ12 | Sections 5–8 |
| 3 | Compare linear and LOWESS fits of planning pauses vs. syntactic tree depth | Section 9 and figures |
| 4 | Translate NJ12 sentences English-to-Hindi with Qwen 2.5 and correlate chrF++ with complexity and human measures | Section 11 |

---

### Files the notebook expects (not included, must be provided)

| File | Source | Purpose |
|---|---|---|
| `CLEAR_corpus_final.xlsx` | CLEAR corpus (Crossley et al., 2023) | Training data; sheet `Data` with an *excerpt* column and a *BT easiness / target* column |
| `NJ12_SessionProps.zip` (or an `NJ12/` folder) | CRITT TPR-DB, study NJ12 | Session XML files with source/target tokens, fixations, modifications and alignments |


### Files the notebook generates (in `WORKDIR`)

| File | Created By | Description |
|---|---|---|
| `clear_features.csv` | Feature extraction | Cached extracted features (avoids re-parsing after a disconnect) |
| `complexity_model.joblib` | Model training | Trained Ridge model pipeline, feature list, and score-scaling bounds |
| `feature_weights.csv`, `feature_weights.png` | Model inspection | Standardized feature weights, relative importance percentages, and bar plot |
| `critt_disaggregated_analysis.csv` | Phase 2 analysis | Sentence-level table of predictors and human cognitive biometrics |
| `critt_with_qwen_eval.csv` | Phase 4 evaluation | Sentence-level translations, multi-reference sets, and chrF++/BLEU scores |
| `test_predictions.csv` | Demo scoring | Output of `score_file()` |
| `figures/fig1_correlation_heatmap.png` | Multi-panel figures | Predictors vs. cognitive biometrics (Report Figure 3) |
| `figures/fig2_translator_normalization.png` | Multi-panel figures | Raw vs. z-normalized source gaze distributions (Report Figure 2) |
| `figures/fig3_crossings_vs_pauses.png` | Multi-panel figures | Alignment crossings vs. planning pauses (Report Figure 5a) |
| `figures/fig4_nonlinear_threshold.png` | Multi-panel figures | Linear vs. LOWESS fit of pauses by tree depth (Report Figure 4) |
| `figures/fig5_case_study_contrast.png` | Multi-panel figures | High- vs. low-effort sentence profile contrast (Report Figure 5b) |
| `figures/fig6_complexity_and_cognition_vs_llm_quality.png` | Multi-panel figures | Complexity and cognitive measures vs. Qwen chrF++ (Report Figure 6) |

---

## 3. Notebook Walkthrough

1. **Install** – Installs required libraries (`stanza`, `wordfreq`, `transformers`, `sacrebleu`, etc.).
2. **Drive, paths, config** – Mounts Google Drive and sets `WORKDIR`, `CSV_PATH`, `MODEL_PATH`, `FEATURE_CACHE`, and the parser `BACKEND` (`'stanza'` by default; `'spacy'` is available).
3. **Parsing and feature extraction** – Stanza dependency parsing extracting 20 features: Zipf frequency (`wordfreq`), syllables, word length, MATTR, lexical density, WordNet polysemy, tree depth, dependency distance, subordinate clauses, passives, commas, punctuation, numbers, and named entities.
4. **Training and feature weights** – `StandardScaler` + `RidgeCV` with 5-fold cross-validation and an 80/20 held-out split; Gradient Boosting as a non-linear reference baseline; refit on full CLEAR data and saved.
5. **Score any text** – `score_text()` computes a calibrated 0–100 complexity score and prints top contributing features; `score_file()` scores an input CSV file.
6. **NJ12 parsing** – Extracts segment-level source/target gaze duration, fixation counts, cognitive planning pauses (inter-keystroke intervals > 1000 ms), keystroke volume, deletion ratio, and word alignment crossings from CRITT SessionProps XMLs.
7. **Within-translator z-score normalization** – Standardizes biometrics per participant session ($z_{ij} = \frac{X_{ij} - \mu_j}{\sigma_j}$) to remove individual typist/reading speed baseline differences, then aggregates to 41 unique sentences.
8. **Disaggregation and correlations** – Computes CLEAR complexity score and fine-grained linguistic predictors for each sentence; calculates Spearman rank correlations against raw and normalized cognitive measures.
9. **Non-linear analysis** – Linear fit vs. LOWESS smoother ($\text{span} = 0.66$) of `z_pauses_per_word` against mean tree depth with $R^2$ evaluation.
10. **Figures 1–5 generation** – Exports multi-panel evaluation plots to `figures/`.
11. **LLM translation quality** – Extracts multi-reference human Hindi translations from `FinalToken`, generates zero-shot translations with `Qwen/Qwen2.5-1.5B-Instruct` (greedy decoding, `max_new_tokens=100`), computes sentence-level chrF++ and BLEU with `sacrebleu`, and generates Figure 6.

### How alignment crossings are computed
For each segment, every pair of source-target alignment links from the `<Alignment>` block of the session XML is compared. A pair counts as one crossing if the source order is reversed in the target ($s_i < s_j$ but $t_i > t_j$, or vice versa). Crossings are summed per segment and averaged across translators for each unique sentence.

---

## 4. Requirements

- Python 3 on **Google Colab** (T4 GPU recommended for Stanza and Qwen inference)
- Packages: `stanza`, `wordfreq`, `nltk`, `scikit-learn`, `scipy`, `joblib`, `tqdm`, `matplotlib`, `seaborn`, `statsmodels`, `openpyxl`, `sacrebleu`, `transformers`, `accelerate`, `bitsandbytes`, `torch`, `pandas`, `numpy`

The first notebook cell installs all necessary packages. WordNet and Stanza English language models download automatically on first run.

---

## 5. How to Run

1. Upload `complexity_model_colab_final.ipynb` to Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Place `CLEAR_corpus_final.xlsx` and `NJ12_SessionProps.zip` in your Google Drive under `MyDrive/CMC_Project/`.
4. If using a different Drive path, adjust `WORKDIR` in Cell 2.
5. Run all cells top to bottom (**Runtime -> Run all**). Intermediate features are cached to Drive, preventing redundant parsing upon reconnects.

---

## 7. References

- **CLEAR Corpus:** Crossley et al. (2023)
- **CRITT TPR-DB:** Carl et al. (2016)
- **Qwen 2.5:** Qwen Team (2024)
- **chrF++:** Popović (2015)

Full bibliographic details are provided in the main project report.
