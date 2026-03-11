import streamlit as st
import pandas as pd
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from transformers import pipeline

st.title("AI Incident Category Classifier")

# --------------------------
# FIXED CATEGORY LIST
# --------------------------

VALID_CATEGORIES = [
"IT - System Access issue",
"IT – System Version issue",
"IT – Data entry handholding",
"IT – Master Data/ mapping issue",
"User - Mapping missing",
"User – Master data delayed input",
"User - Logic changes during ABP",
"User – Master data incorporation in system",
"User – System Knowledge Gap",
"User - Logic mistakes in excel vs system",
"User - Multiple versions issue in excel"
]

# --------------------------
# RULE BASED ENGINE
# --------------------------

RULES = {
"IT - System Access issue":["login","access","permission"],
"IT – Master Data/ mapping issue":["mapping","master data mapping"],
"User - Mapping missing":["mapping missing"],
"User – Master data delayed input":["delayed master data"],
"User - Logic mistakes in excel vs system":["excel mismatch","formula"],
"User - Multiple versions issue in excel":["multiple excel","duplicate file"],
}

# --------------------------
# TEXT COLUMNS
# --------------------------

TEXT_COLUMNS = [
"Ticket Summary",
"Ticket Details",
"Ticket Description",
"Work notes",
"Additional comments",
"Remarks",
"Solution"
]

# --------------------------
# TEXT CLEANING
# --------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\n"," ",text)
    text = re.sub(r"\s+"," ",text)

    return text.strip()

# --------------------------
# COMBINE TEXT
# --------------------------

def combine_columns(df):

    cols = [c for c in TEXT_COLUMNS if c in df.columns]

    if not cols:
        return pd.Series([""]*len(df))

    return df[cols].fillna("").astype(str).agg(" ".join,axis=1)

# --------------------------
# RULE CLASSIFIER
# --------------------------

def rule_classifier(text):

    for cat,words in RULES.items():

        for w in words:

            if w in text:

                return cat

    return None

# --------------------------
# LOAD SUMMARIZER
# --------------------------

@st.cache_resource
def load_summarizer():

    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

summarizer = load_summarizer()

# --------------------------
# SUMMARY GENERATOR
# --------------------------

def generate_summary(text):

    text = str(text)

    if len(text) < 40:
        return text

    try:

        result = summarizer(
            text,
            max_length=40,
            min_length=10,
            do_sample=False
        )

        return result[0]["summary_text"]

    except:

        return text[:120]

# --------------------------
# LOAD TRAINING DATA
# --------------------------

train_df = pd.read_excel("issue_category.xlsx")

train_df.columns = train_df.columns.str.strip()

train_df["combined_text"] = combine_columns(train_df).apply(clean_text)

train_df = train_df[train_df["combined_text"]!=""]

X_train = train_df["combined_text"]
y_train = train_df["Issue category"]

# --------------------------
# ML MODEL
# --------------------------

model = Pipeline([
("tfidf",TfidfVectorizer(
stop_words="english",
ngram_range=(1,2),
min_df=2
)),
("clf",LogisticRegression(
max_iter=3000,
class_weight="balanced"
))
])

model.fit(X_train,y_train)

st.success("Model trained successfully")

# --------------------------
# FILE UPLOAD
# --------------------------

uploaded_file = st.file_uploader(
"Upload Incident Excel File",
type=["xlsx"]
)

if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()

    writer = pd.ExcelWriter(output,engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel,sheet_name=sheet)

        df.columns = df.columns.str.strip()

        df["combined_text"] = combine_columns(df).apply(clean_text)

        predictions=[]
        confidence=[]

        for text in df["combined_text"]:

            rule = rule_classifier(text)

            if rule:

                predictions.append(rule)
                confidence.append(0.97)
                continue

            pred = model.predict([text])[0]

            prob = model.predict_proba([text]).max()

            predictions.append(pred)
            confidence.append(round(prob,3))

        df["Predicted Category"] = predictions
        df["Confidence"] = confidence

        # rewritten summary
        df["Rewritten Summary"] = df["combined_text"].apply(generate_summary)

        df.drop(columns=["combined_text"],inplace=True)

        df.to_excel(writer,sheet_name=sheet,index=False)

    writer.close()

    st.success("Categorization completed")

    st.download_button(
        "Download Categorized File",
        data=output.getvalue(),
        file_name="categorized_incidents.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
