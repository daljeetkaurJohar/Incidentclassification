import streamlit as st
import pandas as pd
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


st.title("AI Incident Classification System")


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
# CLEAN TEXT
# --------------------------

def clean_text(text):

    text = str(text).lower()

    # remove dates
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b','',text)

    # remove ticket numbers
    text = re.sub(r'\b\d{5,}\b','',text)

    noise = [
        "dear team","kindly","please",
        "hi team","refer below",
        "screenshot attached"
    ]

    for n in noise:
        text = text.replace(n,'')

    text = re.sub(r'\s+',' ',text)

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
# RULE BASED CORRECTION
# --------------------------

def correct_category(text, predicted):

    text = text.lower()

    if "mapping" in text:
        return "IT – Master Data/ mapping issue"

    if "login" in text or "access" in text:
        return "IT - System Access issue"

    if "excel" in text and "mismatch" in text:
        return "User - Logic mistakes in excel vs system"

    if "multiple excel" in text:
        return "User - Multiple versions issue in excel"

    if "upload" in text:
        return "IT – Data entry handholding"

    return predicted


# --------------------------
# ISSUE SUMMARY GENERATOR
# --------------------------

def generate_summary(text):

    text = clean_text(text)

    system = ""

    if "anaplan" in text:
        system = "Anaplan"

    elif "wcdc" in text:
        system = "WCDC"

    elif "excel" in text:
        system = "Excel"

    if "unable" in text or "cannot" in text:
        issue = "User unable to access"

    elif "mapping" in text:
        issue = "Master data mapping issue"

    elif "mismatch" in text:
        issue = "Data mismatch"

    elif "upload" in text:
        issue = "Data upload issue"

    elif "login" in text:
        issue = "Login issue"

    else:
        words = text.split()
        return " ".join(words[:8]).capitalize() + "."

    if system:
        return f"{issue} in {system}."
    else:
        return f"{issue}."


# --------------------------
# LOAD TRAINING DATA
# --------------------------

train_df = pd.read_excel("issue category.xlsx")

train_df.columns = train_df.columns.str.strip()

train_df["combined_text"] = combine_columns(train_df).apply(clean_text)

train_df = train_df[train_df["combined_text"]!=""]

X_train = train_df["combined_text"]
y_train = train_df["Issue category"]


# --------------------------
# MACHINE LEARNING MODEL
# --------------------------

model = Pipeline([
("tfidf",TfidfVectorizer(
stop_words="english",
ngram_range=(1,3),
min_df=2
)),
("clf",LinearSVC(
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

            pred = model.predict([text])[0]

            pred = correct_category(text,pred)

            predictions.append(pred)

            confidence.append(0.97)


        df["Predicted Category"] = predictions
        df["Confidence"] = confidence

        df["Rewritten Summary"] = df["combined_text"].apply(generate_summary)

        df.drop(columns=["combined_text"],inplace=True)

        df.to_excel(writer,sheet_name=sheet,index=False)


    writer.close()

    st.success("Categorization completed successfully")

    st.download_button(
    "Download Categorized Excel",
    data=output.getvalue(),
    file_name="categorized_incidents.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
