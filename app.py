import streamlit as st
import pandas as pd
from io import BytesIO
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.title("AI Incident Category Classifier")

# text columns used in training and prediction
TEXT_COLUMNS = [
"Ticket Summary",
"Ticket Details",
"Ticket Description",
"Work notes",
"Additional comments",
"Remarks",
"Solution"
]

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'\n',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text.strip()


def combine_columns(df):

    available = [c for c in TEXT_COLUMNS if c in df.columns]

    if not available:
        return pd.Series([""] * len(df))

    return df[available].fillna("").astype(str).agg(" ".join, axis=1)


# ---------- TRAIN MODEL ----------

train_df = pd.read_excel("issue_category.xlsx")

train_df.columns = train_df.columns.str.strip()

train_df["combined_text"] = combine_columns(train_df).apply(clean_text)

train_df = train_df[train_df["combined_text"] != ""]

X_train = train_df["combined_text"]
y_train = train_df["Issue category"]

model = Pipeline([
("tfidf", TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2),
    min_df=2
)),
("clf", LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
))
])

model.fit(X_train, y_train)

st.success("Model trained using historical categorized tickets")

# ---------- PREDICTION ----------

uploaded_file = st.file_uploader("Upload Incident File", type=["xlsx"])

if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel, sheet_name=sheet)

        df.columns = df.columns.str.strip()

        df["combined_text"] = combine_columns(df).apply(clean_text)

        predictions = model.predict(df["combined_text"])
        probs = model.predict_proba(df["combined_text"])

        confidence = probs.max(axis=1)

        df["Predicted Category"] = predictions
        df["Confidence"] = confidence.round(3)

        # improved rewritten summary
        def summarize(text):

            words = text.split()

            summary = " ".join(words[:20])

            return summary.capitalize()

        df["Rewritten Summary"] = df["combined_text"].apply(summarize)

        df.drop(columns=["combined_text"], inplace=True)

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    st.success("Categorization completed")

    st.download_button(
        "Download Categorized File",
        data=output.getvalue(),
        file_name="categorized_incidents.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
