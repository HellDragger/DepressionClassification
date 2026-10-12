# DepressionClassification

Detecting depression in Reddit posts with classical NLP: NLTK text cleaning, TF-IDF features and a logistic regression classifier. The baseline reaches **~92% accuracy** on a held-out test set.

## Results

`depression_classification.py` trains on 80% of the data and evaluates on the remaining 20% (5,596 posts, `random_state=42`):

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| 0: not depressed | 0.92 | 0.93 | 0.92 | 2,802 |
| 1: depressed | 0.93 | 0.92 | 0.92 | 2,794 |
| **Accuracy** | | | **0.924** | 5,596 |

Reproduced with pandas 3.0, scikit-learn 1.9 and NLTK 3.10 (training plus evaluation takes about 12 s on a laptop). The original report in [`Reports/Classification.pdf`](Reports/Classification.pdf) records 0.9226; the small difference comes from library versions.

## Dataset

[`dataset/Depression_train.csv`](dataset/Depression_train.csv): 27,977 Reddit posts with two columns.

| Column | Description |
| --- | --- |
| `text` | Post body |
| `label` | `1` = indicates depression, `0` = does not |

- **Balanced:** 13,838 depressed posts and 14,139 not depressed.
- **Already cleaned:** every post is lowercase, with no punctuation and no digits. Most stopwords had already been removed upstream.
- **Length:** the median post is 38 words, the mean is 72, and the longest is 2,676 words.
- **Duplicates:** 5 posts appear twice, always with the same label. With the fixed split, 2 test posts also appear in the training set, which has no meaningful effect on the score.

## Project structure

| File | What it does |
| --- | --- |
| [`data_preprocessing.py`](data_preprocessing.py) | Loads the data, reports missing values (there are none), and builds a `processed_text` column: NLTK tokenization, then dropping non-alphabetic tokens and stopwords. Exploration only; nothing is saved. |
| [`depression_classification.py`](depression_classification.py) | The main pipeline: regex cleanup and stopword removal, then an 80/20 split, TF-IDF (`max_features=5000`), `LogisticRegression`, and accuracy plus a classification report. |
| [`trans.py`](trans.py) | Preprocesses the posts and translates the first 1,000 into French with [`Helsinki-NLP/opus-mt-en-fr`](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr) (MarianMT), in batches of 10. |
| [`Reports/Classification.pdf`](Reports/Classification.pdf) | Write-up of the dataset, method, metrics and conclusions. |

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts download the NLTK `stopwords` and `punkt` data on first run. `torch`, `transformers` and `sentencepiece` are only needed for `trans.py`.

## Usage

Run every script from the repository root, because each one reads `dataset/Depression_train.csv` by relative path.

```bash
python depression_classification.py   # train + evaluate (~12 s)
python data_preprocessing.py          # inspect + tokenize (~6 s)
python trans.py                       # EN→FR translation of 1,000 posts (downloads ~300 MB model)
```

## Known issues

- **Preprocessing is largely redundant.** The dataset is already cleaned, so lowercasing and punctuation stripping change almost nothing. In practice, the preprocessing steps only remove leftover stopwords and single-letter words.
- **Dependencies are unpinned** in `requirements.txt`.

## Possible next steps

- Try other linear baselines such as `LinearSVC` and Multinomial Naive Bayes, and add word bigrams to TF-IDF.
- Use cross-validation instead of a single split, and deduplicate the data before splitting.
- Fine-tune a transformer such as DistilBERT or MentalBERT, as the report suggests.

## Disclaimer

This is an academic NLP project. It is not a diagnostic tool and must not be used to assess anyone's mental health.

## License

Apache 2.0. See [LICENSE](LICENSE).
