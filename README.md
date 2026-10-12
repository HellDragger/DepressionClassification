# DepressionClassification

Classifying Reddit posts as depressed / not depressed using classical NLP.

| File | Purpose |
| --- | --- |
| `dataset/Depression_train.csv` | 27,977 posts with `text` and binary `label` |
| `data_preprocessing.py` | Inspect the data and clean text (lowercase, tokenize, drop punctuation/stopwords) |
| `depression_classification.py` | TF-IDF (5,000 features) + logistic regression, 80/20 split |
| `trans.py` | Translate preprocessed posts to French with MarianMT |
| `Reports/Classification.pdf` | Write-up of the classification approach |

## Running

```bash
pip install -r requirements.txt
python depression_classification.py
```
