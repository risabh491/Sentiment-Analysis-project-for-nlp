# 🧠 Sentiment Analysis System

A machine-learning based sentiment analysis system that classifies customer reviews into **Negative, Neutral, or Positive** sentiment.

The project was developed as part of the **CodeAlpha Data Analytics Internship – Task 4: Sentiment Analysis**.

It includes data preprocessing, exploratory analysis, multiple sentiment-model experiments, a reproducible training pipeline, model evaluation, and an interactive Streamlit dashboard.

---

## 🚀 Project Overview

Customer reviews contain valuable information about how users perceive products and services.

This project analyzes review text and automatically determines its sentiment using Natural Language Processing (NLP) and Machine Learning.

### Sentiment Classes

- 🔴 **Negative**
- 🟡 **Neutral**
- 🟢 **Positive**

The final production pipeline uses:

**Text → Preprocessing → TF-IDF → Logistic Regression → Sentiment Prediction**

---

## 🎯 Objectives

- Clean and preprocess customer review text.
- Explore sentiment and rating patterns.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple sentiment-analysis approaches.
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Build a reusable training pipeline.
- Deploy the trained model through an interactive Streamlit dashboard.
- Extract business insights from customer reviews.

---

## 📊 Dataset

The project uses the **Amazon Beauty Reviews** dataset available through Hugging Face.

The original training source contains approximately **2.58 million reviews**.

For this project, a balanced sample was created:

| Sentiment | Reviews |
|-----------|---------:|
| Negative | 5,000 |
| Neutral | 5,000 |
| Positive | 5,000 |
| **Total** | **15,000** |

After removing one duplicate review, the final dataset contains:

**14,999 reviews**

### Dataset Features

- `text_full` — Original review text
- `sentiment` — Sentiment label
- `rating` — Star rating
- `clean_text` — Preprocessed review text

> The dataset was intentionally balanced for machine-learning experimentation. Therefore, the sentiment distribution should not be interpreted as the natural distribution of sentiment among all Amazon reviews.

---

## 🧹 Data Preprocessing

The text preprocessing pipeline includes:

1. Converting text to lowercase.
2. Removing URLs.
3. Removing HTML tags.
4. Normalizing whitespace.
5. Removing duplicate review text.

Example:

```python
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

Negations such as `not`, `don't`, and `didn't` were preserved because they can carry important sentiment information.

---

## 🔬 Exploratory Data Analysis

The project analyzes:

- Sentiment distribution
- Review length
- Star-rating distribution
- Rating vs sentiment relationship
- Frequently occurring words
- Negative phrases
- Positive phrases

### Rating and Sentiment

The dataset shows a strong relationship between star ratings and sentiment labels.

The rating feature was **not used as a model input** because it is closely associated with the target sentiment label and could introduce target leakage.

---

## 🤖 Models Tested

Several approaches were evaluated during experimentation.

### 1. TF-IDF + Logistic Regression

The main machine-learning model.

TF-IDF configuration:

```text
max_features = 10,000
ngram_range = (1, 2)
min_df = 2
```

Train/test split:

```text
80% Training
20% Testing
```

The split was stratified to preserve the class distribution.

---

### 2. Linear SVM

A Linear Support Vector Machine was evaluated as another traditional NLP classifier.

---

### 3. Random Forest

A Random Forest classifier was tested using the TF-IDF representation.

---

### 4. VADER

The rule-based VADER sentiment analyzer was evaluated as a baseline.

---

### 5. TF-IDF + VADER + Logistic Regression

TF-IDF features were combined with VADER sentiment features to test whether rule-based sentiment information could improve the machine-learning model.

---

### 6. DistilBERT

A transformer-based DistilBERT model was also benchmarked.

The DistilBERT experiment used a smaller balanced training subset because the available environment was CPU-only.

Therefore, its result should **not be treated as a directly controlled comparison** with the other models, which used the full training split.

---

## 📈 Model Performance

| Model | Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | **76.07%** |
| TF-IDF + VADER + Logistic Regression | **~76.20%** |
| Linear SVM | 74.10% |
| Random Forest | 74.93% |
| VADER | 53.23% |
| DistilBERT* | 70.80% |

\*DistilBERT was trained using a substantially smaller training subset and for one epoch, so the result is a benchmark rather than a controlled comparison with the other models.

### Final Model

The reproducible production pipeline uses:

**TF-IDF + Logistic Regression**

Test-set performance:

- **Accuracy:** 76.07%
- **Macro F1:** 76.17%

---

## 📋 Classification Performance

Final Logistic Regression model:

| Sentiment | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Negative | 0.74 | 0.75 | 0.75 |
| Neutral | 0.67 | 0.69 | 0.68 |
| Positive | 0.87 | 0.84 | 0.85 |
| **Macro Average** | **0.76** | **0.76** | **0.76** |

---

## 💡 Business Insights

The review analysis revealed several recurring themes.

### Positive Reviews

Positive reviews frequently emphasize:

- Product quality
- Good fit
- Comfort
- Value for money
- Appearance
- Ease of use
- Matching expectations
- Recommendations

Common positive phrases include:

```text
great product
good quality
highly recommend
perfect fit
great price
works great
love dress
fits perfectly
```

### Negative Reviews

Negative reviews frequently mention:

- Poor quality
- Material problems
- Incorrect sizing
- Poor fit
- Value concerns
- Product durability
- Functionality problems
- Difference between expectations and actual product

Common negative phrases include:

```text
waste money
poor quality
runs small
cheap material
didn't work
fell apart
looks cheap
stopped working
```

These patterns can help businesses identify recurring customer pain points and areas for product improvement.

---

## 🖥️ Interactive Dashboard

The project includes a Streamlit dashboard.

The dashboard provides:

### 🔍 Sentiment Analyzer

Users can enter a review and receive:

- Predicted sentiment
- Prediction confidence
- Probability distribution
- Review length

### 📊 Analytics

The dashboard displays:

- Total reviews
- Average rating
- Sentiment distribution
- Rating vs sentiment
- Rating summary

### 💡 Business Insights

The dashboard presents key findings from the review analysis, including recurring positive and negative themes.

---

## 📁 Project Structure

```text
sentiment-intelligence/
│
├── data/
│   ├── raw/
│   │   └── amazon_beauty_reviews_sample.csv
│   │
│   ├── processed/
│   │   └── sentiment_data.csv
│   │
│   └── external/
│
├── models/
│   ├── sentiment_logistic_regression.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── dashboard/
│   └── app.py
│
├── reports/
│   ├── model_metrics.csv
│   └── classification_report.txt
│
├── src/
│
├── setup_project.py
│
└── README.md
```

---

## ⚙️ Installation

Clone or download the project and navigate into the project directory.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn plotly nltk spacy vaderSentiment textblob wordcloud transformers datasets accelerate sentence-transformers torch jupyterlab ipykernel streamlit joblib tqdm openpyxl python-dotenv
```

---

## ▶️ Run the Project

### 1. Build the complete data and model pipeline

Run:

```bash
python setup_project.py
```

This automatically:

- Creates project directories
- Downloads/reads the dataset
- Creates the balanced sample
- Removes duplicates
- Preprocesses review text
- Creates the train/test split
- Trains the TF-IDF + Logistic Regression model
- Evaluates the model
- Saves the trained model
- Saves the TF-IDF vectorizer
- Generates evaluation reports

---

### 2. Launch the dashboard

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

Default address:

```text
http://localhost:8501
```

---

## 🧪 Reproducibility

The project uses fixed random seeds where appropriate:

```python
random_state=42
```

This helps make the sampling, train/test split, and model training reproducible.

The complete pipeline can be regenerated using:

```bash
python setup_project.py
```

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly
- WordCloud

### Machine Learning

- Scikit-learn

### NLP

- TF-IDF
- VADER
- DistilBERT
- Transformers

### Dashboard

- Streamlit

### Development

- VS Code
- Jupyter Notebook
- Python Virtual Environment

---

## 📌 Limitations

- The project uses a balanced sample rather than the complete original dataset.
- The dataset represents Amazon Beauty reviews and may not generalize to every industry.
- The final model uses traditional TF-IDF features rather than a fully fine-tuned transformer model.
- Neutral sentiment is comparatively harder for the classifier to distinguish.
- DistilBERT was evaluated on a smaller CPU-friendly training subset, so its benchmark is not directly comparable to the full-data experiments.

---

## 🔮 Future Improvements

Possible future improvements include:

- Fine-tuning DistilBERT on the complete training dataset.
- Hyperparameter optimization.
- Larger and more diverse datasets.
- Aspect-based sentiment analysis.
- Product-category level sentiment analysis.
- Review-topic clustering.
- Real-time review ingestion.
- Automated business alerts for negative sentiment.
- Deployment to a cloud platform.
- API-based prediction service.

---

## 👨‍💻 Author

**Rachitesh Sharma**

MCA — Artificial Intelligence & Machine Learning

Data Analytics Intern

---

## 📜 Internship

**CodeAlpha Data Analytics Internship**

Task 4 — Sentiment Analysis

---

## ⭐ Project Summary

This project demonstrates an end-to-end NLP and machine-learning workflow:

Raw Reviews
     ↓
Data Sampling
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Text Preprocessing
     ↓
Train/Test Split
     ↓
TF-IDF Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Serialization
     ↓
Streamlit Dashboard
     ↓
Business Insights

The resulting system provides an interactive way to classify customer reviews and explore sentiment patterns using machine learning.
```
