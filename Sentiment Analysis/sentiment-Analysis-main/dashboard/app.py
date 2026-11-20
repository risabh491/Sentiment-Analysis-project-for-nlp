import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1.5rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    opacity: 0.7;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 1rem;
}

.insight-card {
    padding: 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    opacity: 0.5;
    margin-top: 3rem;
    padding: 1rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

model_path = "models/sentiment_logistic_regression.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"
data_path = "data/processed/sentiment_data.csv"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# ============================================================
# LOAD ANALYTICS DATA
# ============================================================

if os.path.exists(data_path):

    df = pd.read_csv(data_path)

else:

    df = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Sentiment Analysis")

    st.divider()

    st.markdown("### Model")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Features:** TF-IDF")
    st.write("**Classes:** 3")
    st.write("**Test Accuracy:** 75.80%")
    st.write("**Macro F1:** 75.91%")

    st.divider()

    st.markdown("### Sentiment")

    st.write("🔴 Negative")
    st.write("🟡 Neutral")
    st.write("🟢 Positive")

    st.divider()

    st.caption(
        "Sentiment Analysis System\n"
        "NLP & Data Analytics Project"
    )


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero-title">
🧠 Sentiment Intelligence
</div>

<div class="hero-subtitle">
AI-powered customer review analysis using Natural Language Processing
and Machine Learning.
</div>
""", unsafe_allow_html=True)

st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🔍 Sentiment Analyzer",
    "📊 Analytics",
    "💡 Business Insights"
])


# ============================================================
# TAB 1 — SENTIMENT ANALYZER
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Analyze a Customer Review</div>',
        unsafe_allow_html=True
    )

    review = st.text_area(
        "Customer Review",
        placeholder=(
            "Example:\n"
            "The product is amazing, great quality and fits perfectly!"
        ),
        height=170,
        label_visibility="collapsed"
    )

    analyze_button = st.button(
        "🔍 Analyze Sentiment",
        use_container_width=True
    )

    if analyze_button:

        if not review.strip():

            st.warning(
                "Please enter a customer review before analyzing."
            )

        else:

            # ----------------------------------------------
            # Transform review
            # ----------------------------------------------

            review_vector = vectorizer.transform([review])

            # ----------------------------------------------
            # Prediction
            # ----------------------------------------------

            prediction = model.predict(review_vector)[0]

            # ----------------------------------------------
            # Probabilities
            # ----------------------------------------------

            probabilities = model.predict_proba(review_vector)[0]

            class_probabilities = dict(
                zip(model.classes_, probabilities)
            )

            confidence = probabilities.max()

            # ----------------------------------------------
            # Result
            # ----------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-title">Analysis Result</div>',
                unsafe_allow_html=True
            )

            if prediction == "positive":

                st.success("🟢 POSITIVE SENTIMENT")

            elif prediction == "negative":

                st.error("🔴 NEGATIVE SENTIMENT")

            else:

                st.warning("🟡 NEUTRAL SENTIMENT")

            # ----------------------------------------------
            # Metrics
            # ----------------------------------------------

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Sentiment",
                    prediction.capitalize()
                )

            with col2:

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "Review Length",
                    f"{len(review.split())} words"
                )

            # ----------------------------------------------
            # Probability distribution
            # ----------------------------------------------

            st.markdown(
                '<div class="section-title">Probability Distribution</div>',
                unsafe_allow_html=True
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "🔴 Negative",
                    f"{class_probabilities['negative'] * 100:.2f}%"
                )

                st.progress(
                    float(class_probabilities["negative"])
                )

            with col2:

                st.metric(
                    "🟡 Neutral",
                    f"{class_probabilities['neutral'] * 100:.2f}%"
                )

                st.progress(
                    float(class_probabilities["neutral"])
                )

            with col3:

                st.metric(
                    "🟢 Positive",
                    f"{class_probabilities['positive'] * 100:.2f}%"
                )

                st.progress(
                    float(class_probabilities["positive"])
                )

            # ----------------------------------------------
            # Review
            # ----------------------------------------------

            st.markdown(
                '<div class="section-title">Analyzed Review</div>',
                unsafe_allow_html=True
            )

            st.info(review)


# ============================================================
# TAB 2 — ANALYTICS
# ============================================================

with tab2:

    st.header("📊 Customer Sentiment Analytics")

    if df is None:

        st.error(
            "Analytics dataset not found. "
            "Run the export cell from Notebook 1 first."
        )

    else:

        # ----------------------------------------------
        # KPI CARDS
        # ----------------------------------------------

        total_reviews = len(df)

        avg_rating = df["rating"].mean()

        positive_count = (
            df["sentiment"] == "positive"
        ).sum()

        negative_count = (
            df["sentiment"] == "negative"
        ).sum()

        neutral_count = (
            df["sentiment"] == "neutral"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Reviews",
                f"{total_reviews:,}"
            )

        with col2:

            st.metric(
                "Average Rating",
                f"{avg_rating:.2f} / 5"
            )

        with col3:

            st.metric(
                "Positive Reviews",
                f"{positive_count:,}"
            )

        with col4:

            st.metric(
                "Negative Reviews",
                f"{negative_count:,}"
            )

        st.divider()

        # ----------------------------------------------
        # SENTIMENT DISTRIBUTION
        # ----------------------------------------------

        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
            .reindex(
                ["negative", "neutral", "positive"]
            )
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sentiment_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Customer Sentiment Distribution"
        )

        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Reviews")

        ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------
        # RATING VS SENTIMENT
        # ----------------------------------------------

        st.subheader(
            "Sentiment Distribution by Star Rating"
        )

        rating_counts = pd.crosstab(
            df["rating"],
            df["sentiment"]
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        rating_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title(
            "Sentiment Distribution by Star Rating"
        )

        ax.set_xlabel("Star Rating")
        ax.set_ylabel("Number of Reviews")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        plt.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------------
        # RATING SUMMARY
        # ----------------------------------------------

        st.subheader(
            "Rating Statistics by Sentiment"
        )

        rating_summary = (
            df.groupby("sentiment")["rating"]
            .agg(
                ["count", "mean", "median", "min", "max"]
            )
            .round(2)
        )

        st.dataframe(
            rating_summary,
            use_container_width=True
        )


# ============================================================
# TAB 3 — BUSINESS INSIGHTS
# ============================================================

with tab3:

    st.header("💡 Business Insights")

    st.markdown("""
### Key Customer Insights

**1. Product quality matters**

Positive reviews frequently contain phrases such as
**"good quality", "great quality", and "high quality"**,
while negative reviews include **"poor quality"** and
**"cheap material"**.

---

**2. Fit and sizing are important**

Negative reviews frequently mention sizing and fit problems,
including **"runs small", "way small", "way big",
and "doesn't fit"**.

Positive reviews contain phrases such as
**"perfect fit", "fits perfectly", and "great fit"**.

---

**3. Value for money affects customer satisfaction**

Negative reviews contain phrases such as
**"waste money"** and **"worth money"**.

Positive reviews include phrases such as
**"great price", "good price", and "great buy"**.

---

**4. Product functionality and durability matter**

Negative reviews include phrases such as
**"didn't work", "fell apart", and "stopped working"**.

Positive reviews include phrases such as
**"works great"** and **"easy use"**.

---

**5. Customer expectations matter**

The phrase analysis shows references to whether products
match their expected appearance, including phrases around
**"like picture"**.

This suggests that expectation mismatch can contribute
to negative customer experiences.
""")

    st.divider()

    st.subheader("Model Performance")

    performance = pd.DataFrame({
        "Model": [
            "TF-IDF + Logistic Regression",
            "TF-IDF + VADER + Logistic Regression",
            "Linear SVM",
            "Random Forest",
            "VADER",
            "DistilBERT"
        ],
        "Accuracy": [
            0.7580,
            0.7620,
            0.7410,
            0.7493,
            0.5323,
            0.7080
        ]
    })

    st.dataframe(
        performance.style.format({
            "Accuracy": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Note: the DistilBERT experiment used a smaller "
        "resource-constrained training subset, so its accuracy "
        "is not a directly controlled comparison with the "
        "full-data experiments."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">
Sentiment Analysis System • NLP • Machine Learning • Data Analytics
</div>
""", unsafe_allow_html=True)