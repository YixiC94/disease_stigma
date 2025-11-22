# Disease Stigma Pipeline

This repository accompanies the paper ["The Stigma of Diseases: Unequal Burden, Uneven Decline"](https://osf.io/preprints/socarxiv/7nm9x/) by Rachel Kahn Best and Alina Arseniev-Koehler (forthcoming in *American Sociological Review*).

Using word-embedding methods, the project measures stigma for 106 health conditions across 4.7 million news articles from 1980–2018. Code in this repository prepares corpora, trains models, validates embeddings, and aggregates stigma indices. The codebase has been reorganized into functional directories (data preparation, training, validation, analysis) with shared paths centralized in `config/`.

## Directory structure

```
config/                      # Shared configuration (path helpers)
data_prep/                   # Corpus preparation and lexicon utilities
training/                    # Training scripts for phrasers and Word2Vec models
validation/                  # Model and dimension validation utilities
analysis/                    # Aggregation, plotting, and scoring scripts

data/                        # User-provided corpora and prepared pickles (not tracked)
reference_data/              # Reference CSV/text inputs (lexicons, traits, questions)
outputs/
  models/                    # Default location for trained models and intermediates
  results/                   # Default location for aggregated CSVs and plots
notebooks/                   # Reserved for exploratory notebooks
```

Key reference files live in `reference_data/`:

* `Stigma_WordLists.csv`
* `Disease_list_5.12.20_uncorrupted.csv`
* `updated_personality_trait_list.csv`
* `questions_words_pasted.txt`
* `Final_Search_SymptomsDiseasesList.txt`

An example aggregated output is provided at `outputs/results/stigmaindex_aggregated_temp_92CI.csv`.

## Script overview

### Data preparation
* `data_prep/prepare_corpus_from_csv.py`: Convert news article CSVs into yearly pickle files (`NData_<year>/all<year>bodytexts_regexeddisamb_listofarticles`) expected by the training scripts.
* `data_prep/build_lexicon_stigma.py`: Build stigma-related lexicons used across training, validation, and analysis steps.

### Training
* `training/training_phraser.py`: Train phrasers on text data for configurable time windows. Supports two modes:
  - Single window: Specify `--start-year` and `--year-interval` to train one phraser for that interval.
  - Batch mode: Add `--end-year` to automatically train phrasers for all intervals from `start-year` to `end-year` (inclusive), each of length `year-interval`.
* `training/training_w2v_booted.py`: Train bootstrapped Word2Vec models for configurable time windows. Supports two modes:
  - Single window: Specify `--start-year` and `--year-interval` to train models for one interval.
  - Batch mode: Add `--end-year` to automatically train models for all intervals from `start-year` to `end-year` (inclusive), each of length `year-interval`. Each interval will use its corresponding phraser model.

### Validation
* `validation/validating_overall_w2v_models.py`: Validate an overall Word2Vec model on the WordSim-353 test set and the Google analogy test (`reference_data/questions_words_pasted.txt`).
* `validation/validating_dimensions_in_bootstraps.py`: Perform cross-validation for each of the four stigma dimensions, compute cosine similarities, and list the most/least similar words. Requires `data_prep/build_lexicon_stigma.py` and `analysis/dimension_stigma.py`.

### Analysis and aggregation
* `analysis/write_stigma_scores.py`: Compute four stigma scores plus a medicalization score for each disease across models/time windows and write CSVs. Depends on `data_prep/build_lexicon_stigma.py`, `analysis/dimension_stigma.py`, and inputs such as `reference_data/updated_personality_trait_list.csv`, `reference_data/Stigma_WordLists.csv`, and `reference_data/Disease_list_5.12.20_uncorrupted.csv`.
* `analysis/aggregating_stigma_index.py`: Aggregate bootstrapped scores across time windows to produce mean and 92% confidence intervals for each disease’s stigma index (writes a consolidated CSV, e.g., `stigmaindex_aggregated_temp_92CI.csv`).
* `analysis/aggregating_bootstraps.py`: Aggregate bootstrapped scores per dimension and time window, emitting one CSV per dimension.
* `analysis/word_counts.py`: Compute per-disease mention counts with confidence intervals. Requires `reference_data/Disease_list_5.12.20_uncorrupted.csv` and currently contains legacy hard-coded paths.
* `analysis/plotting_bootstrapped.py`: Visualize stigma scores of diseases by group and time (expects an aggregated CSV such as `stigmaindex_aggregated_temp_92CI.csv`).

## Dependencies

The pipeline relies on the following Python packages:

* `gensim`
* `pandas`
* `numpy`
* `matplotlib`
* `nltk`
* `scikit-learn`

Install them with `pip install -r requirements.txt` (if available) or `pip install gensim pandas numpy matplotlib nltk scikit-learn`.

## Shared path configuration

`config/path_config.py` centralizes path handling. Scripts import it via `from config.path_config import add_path_arguments, build_path_config` and accept consistent CLI flags:

* `--raw-data-root`: Base directory containing `NData_<year>` folders with article pickles (default: `data/preprocessed`).
* `--contemp-data-root`: Optional override for `ContempData_<year>` folders (falls back to `--raw-data-root`).
* `--modeling-dir-base`: Base directory for modeling outputs (default: `outputs/models`). Phraser models will be saved in `<base>/phrasers`, W2V models in `<base>/BootstrappedModels`.
* `--results-dir`: Where result CSVs and plots are written (default: `outputs/results`).
* `--analyses-dir`: Base directory for ancillary files (default: `analysis`).
* `--lexicon-path`, `--disease-list-path`, `--personality-traits-path`: Paths to the bundled CSV inputs in `reference_data/`.

## Typical workflow

1. **Prepare corpus pickles** (from CSV input):
   ```bash
   python data_prep/prepare_corpus_from_csv.py \
     --csv-path /path/to/articles.csv \
     --text-column Text --title-column title --date-column Date \
     --default-year 2010 --output-root data/preprocessed --write-manifest
   ```

2. **Train phrase model(s) for time windows**:
   - Single window:
     ```bash
     python training/training_phraser.py \
       --start-year 1992 --year-interval 3 --raw-data-root data/preprocessed --modeling-dir-base outputs/models
     ```
   - Batch mode (multiple windows):
     ```bash
     python training/training_phraser.py \
       --start-year 1980 --year-interval 3 --end-year 1991 --raw-data-root data/preprocessed --modeling-dir-base outputs/models
     ```
   This will train all intervals from 1980–1982, 1983–1985, … up to 1991.

3. **Train bootstrapped Word2Vec models**:
   - Single window:
     ```bash
     python training/training_w2v_booted.py \
       --start-year 1992 --year-interval 3 --boots 25 \
       --min-count 50 --window 10 --vector-size 300 \
       --raw-data-root data/preprocessed --modeling-dir-base outputs/models
     ```
   - Batch mode (multiple windows):
     ```bash
     python training/training_w2v_booted.py \
       --start-year 1980 --year-interval 3 --end-year 1991 --boots 25 \
       --min-count 50 --window 10 --vector-size 300 \
       --raw-data-root data/preprocessed --modeling-dir-base outputs/models
     ```
   This will train all intervals from 1980–1982, 1983–1985, … up to 1991.
   - `--model-prefix` is optional; when omitted, a name is auto-generated (e.g., `CBOW_300d__win10_min50_iter3`) and stored in `BootstrappedModels/<years>/training_manifest.json` along with training parameters.
   - For mock/testing runs you can lower `--min-count` (e.g., 1) and/or supply a different prefix to keep outputs distinct from official runs.

4. **Validate models** (run after training to catch issues early):
  * Overall analogies: `python validation/validating_overall_w2v_models.py --model-path outputs/models/<model>`
  * Dimension quality: `python validation/validating_dimensions_in_bootstraps.py --modeling-dir-base outputs/models --year-interval 3 --model-prefix <prefix>`
    - You may omit `--model-prefix`; the script will read `training_manifest.json` to discover it.

5. **Compute stigma scores per dimension**:
   ```bash
   python analysis/write_stigma_scores.py \
     --modeling-dir-base outputs/models --results-dir outputs/results \
     --start-year 1980 --year-interval 3 --end-year 2016 --boots 25 \
     --plotting-groups neurodevelopmental
   ```
   - You may omit `--model-prefix`; the script will read `training_manifest.json` for each interval.
   - For mock data you can use fewer `--boots` and a lower `--lexicon-min-count`.

6. **Aggregate and plot**:
   ```bash
   python analysis/aggregating_stigma_index.py \
     --results-dir outputs/results \
     --start-year 1980 --end-year 2016 --year-interval 3 \
     --dimensions negpostraits disgust danger impurity

   python analysis/plotting_bootstrapped.py \
     --dimension stigmaindex --results-dir outputs/results
   ```
   - Aggregation now accepts `--start-year/--end-year/--year-interval` to match what you trained, and skips missing temp files with a warning.
   - For mock runs, limit the years and boots you actually produced (e.g., `--start-year 1980 --end-year 1980 --boots 1`) to avoid missing-file noise.

## Notes and outstanding manual adjustments

* `analysis/word_counts.py` still contains legacy, hard-coded paths and should be updated before use.
* Validation and analysis scripts assume trained models and bootstrapped artifacts exist in `outputs/models`.
* Raw LexisNexis corpora are not distributed; use your own data with the provided preprocessing script.

## Historical context

Full methodological details and empirical findings are described in the linked preprint and its appendices.
