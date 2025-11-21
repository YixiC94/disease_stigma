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

data/                        # Reference CSV/text inputs (lexicons, traits, questions)
outputs/
  models/                    # Default location for trained models and intermediates
  results/                   # Default location for aggregated CSVs and plots
notebooks/                   # Reserved for exploratory notebooks
```

Key reference files live in `data/`:

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
* `training/TrainingPhraser_CleanedUp.py`: Train a phraser on text data from a given time window (one phraser per window).
* `training/TrainingW2V_Booted_CleanedUp.py`: Train bootstrapped Word2Vec models using the phrasers trained for each time window.

### Validation
* `validation/Validating_OverallW2VModels_CleanedUp.py`: Validate an overall Word2Vec model on the WordSim-353 test set and the Google analogy test (`data/questions_words_pasted.txt`).
* `validation/Validating_Dimensions_in_Bootstraps_CleanedUp.py`: Perform cross-validation for each of the four stigma dimensions, compute cosine similarities, and list the most/least similar words. Requires `data_prep/build_lexicon_stigma.py` and `analysis/dimension_stigma.py`.

### Analysis and aggregation
* `analysis/WriteStigmaScores_CleanedUp.py`: Compute four stigma scores plus a medicalization score for each disease across models/time windows and write CSVs. Depends on `data_prep/build_lexicon_stigma.py`, `analysis/dimension_stigma.py`, and inputs such as `data/updated_personality_trait_list.csv`, `data/Stigma_WordLists.csv`, and `data/Disease_list_5.12.20_uncorrupted.csv`.
* `analysis/AggregatingStigmaScores_StigmaIndex_CleanedUp.py`: Aggregate bootstrapped scores across time windows to produce mean and 92% confidence intervals for each disease’s stigma index (writes a consolidated CSV, e.g., `stigmaindex_aggregated_temp_92CI.csv`).
* `analysis/AggregatingBootstraps_CleanedUp.py`: Aggregate bootstrapped scores per dimension and time window, emitting one CSV per dimension.
* `analysis/WordCounts_CleanedUp.py`: Compute per-disease mention counts with confidence intervals. Requires `data/Disease_list_5.12.20_uncorrupted.csv` and currently contains legacy hard-coded paths.
* `analysis/PlottingBootstrapped_CleanedUp.py`: Visualize stigma scores of diseases by group and time (expects an aggregated CSV such as `stigmaindex_aggregated_temp_92CI.csv`).

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

* `--raw-data-root`: Base directory containing `NData_<year>` folders with article pickles (default: `data/raw`).
* `--contemp-data-root`: Optional override for `ContempData_<year>` folders (falls back to `--raw-data-root`).
* `--modeling-dir`: Intermediate artifacts such as bigrams, bootstraps, and embeddings (default: `outputs/models`).
* `--results-dir`: Where result CSVs and plots are written (default: `outputs/results`).
* `--analyses-dir`: Base directory for ancillary files (default: `analysis`).
* `--lexicon-path`, `--disease-list-path`, `--personality-traits-path`: Paths to the bundled CSV inputs in `data/`.

## Typical workflow

1. **Prepare corpus pickles** (from CSV input):
   ```bash
   python data_prep/prepare_corpus_from_csv.py \
     --csv-path /path/to/articles.csv \
     --text-column Text --title-column title --date-column Date \
     --default-year 2010 --output-root data/raw --write-manifest
   ```

2. **Train phrase model for a 3-year window**:
   ```bash
   python training/TrainingPhraser_CleanedUp.py \
     --year 1992 --raw-data-root data/raw --modeling-dir outputs/models
   ```

3. **Train bootstrapped Word2Vec models**:
   ```bash
   python training/TrainingW2V_Booted_CleanedUp.py \
     --year 1992 --boots 25 --model-prefix CBOW_300d__win10_min50_iter3 \
     --raw-data-root data/raw --modeling-dir outputs/models
   ```

4. **Compute stigma scores per dimension**:
   ```bash
   python analysis/WriteStigmaScores_CleanedUp.py \
     --modeling-dir outputs/models --results-dir outputs/results
   ```

5. **Aggregate and plot**:
   ```bash
   python analysis/AggregatingStigmaScores_StigmaIndex_CleanedUp.py \
     --modeling-dir outputs/models --results-dir outputs/results

   python analysis/PlottingBootstrapped_CleanedUp.py \
     --dimension stigmaindex --results-dir outputs/results
   ```

6. **Validate models** (examples):
   * Overall analogies: `python validation/Validating_OverallW2VModels_CleanedUp.py --model-path outputs/models/<model>`
   * Dimension quality: `python validation/Validating_Dimensions_in_Bootstraps_CleanedUp.py --modeling-dir outputs/models`

## Notes and outstanding manual adjustments

* `analysis/WordCounts_CleanedUp.py` still contains legacy, hard-coded paths and should be updated before use.
* Validation and analysis scripts assume trained models and bootstrapped artifacts exist in `outputs/models`.
* Raw LexisNexis corpora are not distributed; use your own data with the provided preprocessing script.

## Historical context

Full methodological details and empirical findings are described in the linked preprint and its appendices.
