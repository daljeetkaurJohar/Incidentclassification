import streamlit as st
import pandas as pd
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

st.title("AI Incident Classification System")

# ------------------------------------------------
# TEXT COLUMNS USED
# ------------------------------------------------

TEXT_COLUMNS = [
"Ticket Summary",
"Ticket Details",
"Ticket Description",
"Work notes",
"Additional comments",
"Remarks",
"Solution",
"Support required for issues' closure",
"Any reason for delay",
"Area Category",
"Planning Area"
]

# ------------------------------------------------
# CLEAN TEXT
# ------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b','',text)
    text = re.sub(r'\b\d{5,}\b','',text)

    noise = [
        "dear team","kindly","please",
        "hi team","refer below",
        "refer the below","screenshot attached"
    ]

    for n in noise:
        text = text.replace(n,'')

    text = re.sub(r'\s+',' ',text)

    return text.strip()

# ------------------------------------------------
# COMBINE TEXT FROM ALL COLUMNS
# ------------------------------------------------

def combine_columns(df):

    cols = [c for c in TEXT_COLUMNS if c in df.columns]

    if not cols:
        return pd.Series([""]*len(df))

    return df[cols].fillna("").astype(str).agg(" ".join,axis=1)

# ------------------------------------------------
# RULE BASED CLASSIFIER
# ------------------------------------------------

def rule_classifier(text):

    text = text.lower()

    # ACCESS ISSUE
    if any(k in text for k in [
        "login","log in","unable to login",
        "unable to access","cannot access",
        "access denied","permission",
        "authorization","access request"
    ]):
        return "IT - System Access issue"

    # SYSTEM LINKAGE
    if any(k in text for k in [
        "not flowing","not reflecting",
        "not updating","not syncing",
        "not coming","not showing",
        "transfer not reflected"
    ]):
        return "IT - System linkage issue"

    # VERSION ISSUE
    if any(k in text for k in [
        "version","upgrade","patch"
    ]):
        return "IT – System Version issue"

    # DATA ENTRY HELP
    if any(k in text for k in [
        "how to","data entry",
        "upload help","how do i"
    ]):
        return "IT – Data entry handholding"

    # MASTER DATA MAPPING
    if "mapping" in text:
        return "IT – Master Data/ mapping issue"

    # USER MAPPING MISSING
    if "mapping missing" in text:
        return "User - Mapping missing"

    # MASTER DATA DELAY
    if any(k in text for k in [
        "delayed master data","master data delay"
    ]):
        return "User – Master data delayed input"

    # LOGIC CHANGE
    if "logic change" in text:
        return "User - Logic changes during ABP"

    # MASTER DATA ADDITION
    if any(k in text for k in [
        "add master data","new master data"
    ]):
        return "User – Master data incorporation in system"

    # EXCEL LOGIC
    if any(k in text for k in [
        "excel mismatch","cost sheet","excel logic"
    ]):
        return "User - Logic mistakes in excel vs system"

    # MULTIPLE FILE
    if any(k in text for k in [
        "multiple excel","multiple version"
    ]):
        return "User - Multiple versions issue in excel"

    # KNOWLEDGE GAP
    if any(k in text for k in [
        "training","need help","how to use"
    ]):
        return "User – System Knowledge Gap"

    return None

# ------------------------------------------------
# SUMMARY GENERATOR
# ------------------------------------------------

def generate_summary(text):

    text = clean_text(text)

    if "login" in text or "access" in text:
        return "User unable to login or access the system."

    if "mapping" in text:
        return "Master data mapping issue detected."

    if "mismatch" in text:
        return "Data mismatch observed in system."

    if "upload" in text:
        return "User facing issue while uploading data."

    if "report" in text:
        return "Report not visible or incorrect in system."

    words = text.split()

    return " ".join(words[:10]).capitalize() + "."


# ------------------------------------------------
# TRAIN ML MODEL (fallback only)
# ------------------------------------------------

train_df = pd.read_excel("issue_category.xlsx")

train_df.columns = train_df.columns.str.strip()

train_df["combined_text"] = combine_columns(train_df).apply(clean_text)

train_df = train_df[train_df["combined_text"]!=""]

X_train = train_df["combined_text"]
y_train = train_df["Issue category"]

model = Pipeline([
("tfidf",TfidfVectorizer(
stop_words="english",
ngram_range=(1,3),
min_df=2
)),
("clf",LinearSVC(class_weight="balanced"))
])

model.fit(X_train,y_train)

st.success("Model trained successfully")

# ------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------

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
                confidence.append(1.0)

            else:
                pred = model.predict([text])[0]
                predictions.append(pred)
                confidence.append(0.85)

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
