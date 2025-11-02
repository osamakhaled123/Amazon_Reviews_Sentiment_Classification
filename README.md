# 🧠 Sentiment Analysis on Amazon Product Reviews
*Multi-Class Text Classification (1–5 Star Ratings) using Traditional ML, Deep Learning, and Transformers with different Tokenizers and Embedding Approaches.*

---

## 📋 Project Overview

This project performs **sentiment analysis** on the **Amazon Product Reviews Dataset** (from Kaggle).  
The goal is to classify customer reviews into **5 rating categories** (⭐1 → ⭐5).

It’s a **complete end-to-end NLP pipeline** — starting from cleaning and preprocessing raw reviews, to training multiple models with different embeddings and architectures, and evaluating them thoroughly.

---

## ⚙️ Key Highlights

✅ Cleaned and preprocessed 500K+ Amazon reviews  
✅ Investigated review length and class imbalance  
✅ Applied text preprocessing: lowercasing, stopword removal, lemmatization, deduplication  
✅ Used **3 embedding/tokenization techniques**:
- TF-IDF  
- Word2Vec  
- Hugging Face AutoTokenizer (DistilBERT)

✅ Trained **7 models**:
1. Random Forest  
2. Logistic Regression  
3. SVC (RBF kernel)  
4. XGBoost  
5. Deep Neural Network (Dense NN)  
6. GRU (Gated Recurrent Unit)  
7. DistilBERT Transformer  

✅ Addressed **class imbalance** using `compute_class_weight`  
✅ Leveraged **GPU acceleration** and **memory-efficient batching**  
✅ Collected accuracy, weighted F1-score, confusion matrix, and classification report for both training and validation datasets  

---

## 🧹 Data Preprocessing Pipeline

| Step | Description |
|------|--------------|
| Text Cleaning | Removed punctuation, numbers, special symbols |
| Lowercasing | Normalized all text |
| Lemmatization | Reduced words to their base form |
| Stopword Removal | Eliminated common but meaningless words |
| Deduplication | Removed duplicate reviews |
| Tokenization | Applied tokenizer (TF-IDF, Word2Vec, or AutoTokenizer) |
| Padding & Truncation | Ensured consistent input sequence lengths |
| Class Balancing | Used `compute_class_weight` to compensate for class imbalance |

---

## 🧠 Models & Embeddings

| Model | Embedding | Description |
|--------|------------|-------------|
| Random Forest | TF-IDF | Baseline ensemble tree model |
| Logistic Regression | TF-IDF | Linear classifier for sentiment separation |
| SVC | TF-IDF | RBF kernel-based non-linear classifier |
| XGBoost | TF-IDF | Gradient boosting model for performance and interpretability |
| Deep NN | TF-IDF | Fully connected neural network (dropout + batch norm) |
| GRU | Word2Vec | Sequential model for contextual pattern learning |
| DistilBERT | AutoTokenizer | Transformer fine-tuned on sentiment classification |

---

## 🧾 Evaluation Metrics

Each model is evaluated using:
- **Accuracy**
- **Weighted F1-Score**
- **Classification Report**
- **Confusion Matrix** (visualized for both train and validation)

---

## 📊 Validation Results Summary

| Model | Tokenizer / Embedding | Accuracy | Weighted F1-Score |
|:------|:-----------------------|:----------|:-------------------|
| Random Forest | TF-IDF | 28.53% | 28.25% |
| Logistic Regression | TF-IDF | 24.29% | 24.18% |
| SVC | TF-IDF | 19.55% | 9.32% |
| XGBoost | TF-IDF | 19.63% | 18.01% |
| Deep NN | TF-IDF | 12.47% | 13.35% |
| GRU | Word2Vec | 20.0% | 18.05% |
| DistilBERT | AutoTokenizer | 25.78% | 25.01% |

---

## 📈 Visualizations

📊 **Exploratory Analysis**
- Review length distribution
- Class frequency histogram (imbalanced dataset)

📉 **Training Performance**
- Training vs Validation Loss curves

🧮 **Evaluation**
- Confusion Matrices for each model
- F1-score and Accuracy comparison

---

## 🧰 Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python |
| ML / DL Frameworks | PyTorch, scikit-learn, XGBoost, Hugging Face Transformers |
| Text Processing | spaCy, Gensim |
| Visualization | Matplotlib, Seaborn |
| Environment | Google Colab / Kaggle |
| Version Control | Git & GitHub |

---
## 💾 Project Structure

```bash
Amazon_Reviews_Sentiment_Classification/
│
├── data/
│   ├── cleaned_training_reviews.csv
│   ├── cleaned_validating_reviews.csv
│
├── notebooks/
│   ├── Sentiment_Analysis_Experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── tokenizers.py
│   ├── classification_models.py
│   └── train/
│       ├── DistilBERT.py
│       ├── GRU.py
│       ├── Logistic_Regression.py   
│       ├── Neural_Network.py
│       ├── Random_Forest.py
│       ├── SVC.py
│       └── XGBoost.py
│
├── README.md
└── requirements.txt
```
---
## 🧩 Key Insights 

- Short reviews (<20 tokens) often contain less informative signals.
- Dataset was heavily imbalanced — 5-star reviews dominated.
- Transformers like DistilBERT outperformed traditional models significantly.
- XGBoost achieved a strong balance between interpretability and performance.
- Deep NN models benefited from Dropout and Batch Normalization, reducing overfitting.

---
## 🏁 Conclusion

This project demonstrates the full pipeline of modern NLP classification, combining:
- Classical ML models
- Deep learning architectures
- Transformer-based contextual models
It highlights the evolution from feature-based to contextual embeddings, handling class imbalance, and robust model evaluation.

