# 🎬 Sentiment Analysis on IMDB Reviews using RNN

> An end-to-end NLP pipeline that classifies movie reviews as **positive or negative** using a Recurrent Neural Network built with PyTorch — achieving **85.16% test accuracy** on 50,000 IMDB reviews.

---

## 📌 Project Highlights

- Processed and cleaned **~50,000 real-world IMDB movie reviews** through an 8-step NLP preprocessing pipeline
- Applied **TF-IDF vectorization** (top 5,000 features) to convert text into numerical representations
- Designed and trained a custom **RNN architecture** in PyTorch for binary sentiment classification
- Achieved **85.16% accuracy** on the held-out test set

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| Source | [IMDB Movie Reviews Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |
| Total Samples | 50,000 (after dedup: 49,582) |
| Classes | Positive / Negative (binary) |
| Train / Test Split | 80% / 20% → 39,665 / 9,917 samples |

---

## 🔄 NLP Preprocessing Pipeline

Raw text reviews go through 8 sequential cleaning steps before being fed to the model:

```
Raw Review Text
      │
      ▼
1. Lowercase Conversion       → standardize all characters
      │
      ▼
2. URL Removal                → strip http/https links (regex)
      │
      ▼
3. Punctuation Removal        → keep only alphanumeric & whitespace
      │
      ▼
4. HTML Tag Removal           → remove <br />, <div>, etc.
      │
      ▼
5. Stopword Removal           → drop common words (NLTK English corpus)
      │
      ▼
6. Stemming                   → reduce words to root form (PorterStemmer)
      │
      ▼
7. Label Encoding             → positive → 1, negative → 0
      │
      ▼
8. TF-IDF Vectorization       → 5,000-feature sparse matrix → dense tensor
      │
      ▼
  Model-Ready Input
```

---

## 🏗️ Model Architecture

```
Input (batch_size × 1 × 5000)   ← TF-IDF features as a single timestep
        │
        ▼
  RNN Layer
  ├─ input_size  = 5000
  ├─ hidden_size = 128
  └─ num_layers  = 1
        │
        ▼
  Last Hidden State  (batch_size × 128)
        │
        ▼
  Linear(128 → 1)
        │
        ▼
  Sigmoid Activation  →  probability ∈ [0, 1]
        │
        ▼
  Threshold @ 0.5  →  Positive / Negative
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Test Accuracy | **85.16%** |
| Loss Function | Binary Cross-Entropy (BCELoss) |
| Training Epochs | 10 |
| Optimizer | Adam (default lr = 0.001) |
| Batch Size | 64 |

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch |
| NLP / Text Processing | NLTK (tokenization, stopwords, PorterStemmer) |
| Feature Engineering | scikit-learn TF-IDF Vectorizer |
| Data Handling | pandas, NumPy |
| Label Encoding | scikit-learn LabelEncoder |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/rnn-sentiment-analysis.git
cd rnn-sentiment-analysis
```

### 2. Install dependencies
```bash
pip install torch pandas scikit-learn nltk
```

### 3. Download NLTK resources
```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

### 4. Add the dataset
Download `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project root.

### 5. Run the notebook
```bash
jupyter notebook RNN_for_sentimentanalysis.ipynb
```

---

## 📁 Project Structure

```
rnn-sentiment-analysis/
│
├── RNN_for_sentimentanalysis.ipynb   # Full pipeline: preprocessing → training → evaluation
├── IMDB Dataset.csv                  # Raw dataset (download separately from Kaggle)
└── README.md
```

---

## 🔮 Potential Improvements

- Replace vanilla RNN with **LSTM or GRU** to better handle long-range dependencies and vanishing gradients
- Use **word embeddings** (Word2Vec, GloVe) instead of TF-IDF for richer semantic representations
- Apply **bidirectional RNN** to capture context from both directions in a review
- Add **dropout regularization** to reduce overfitting
- Experiment with **transformer-based models** (e.g., BERT) for state-of-the-art NLP performance

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
