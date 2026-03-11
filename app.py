import streamlit as st
import pandas as pd
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


st.title("AI Incident Classification System")


# ---------------------------------------
# TEXT COLUMNS USED FOR CONTEXT
# ---------------------------------------

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


# ---------------------------------------
# CLEAN TEXT
# ---------------------------------------

def clean_text(text):

    text = str(text).lower()

    # remove dates
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b','',text)

    # remove long ids
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


# ---------------------------------------
# COMBINE TEXT COLUMNS
# ---------------------------------------

def combine_columns(df):

    available = [c for c in TEXT_COLUMNS if c in df.columns]

    if not available:
        return pd.Series([""]*len(df))

    return df[available].fillna("").astype(str).agg(" ".join,axis=1)


# ---------------------------------------
# LEARN KEYWORDS FROM TRAINING DATA
# ---------------------------------------

def learn_category_keywords(df):

    keyword_dict = {}

    for category in df["Issue category"].unique():

        subset = df[df["Issue category"] == category]

        text = " ".join(subset["combined_text"])

        words = text.split()

        freq = {}

        for w in words:

            if len(w) < 4:
                continue

            freq[w] = freq.get(w,0) + 1

        sorted_words = sorted(freq,key=freq.get,reverse=True)

        keyword_dict[category] = sorted_words[:10]

    return keyword_dict


# ---------------------------------------
# STRONG RULE ENGINE
# ---------------------------------------

def rule_classifier(text):

    text = text.lower()

    # SYSTEM ACCESS
    if any(k in text for k in [
        "login","log-in","unable to login",
        "unable to open","cannot open",
        "access","permission","authorization"
    ]):
        return "IT - System Access issue"

    # SYSTEM LINKAGE
    if any(k in text for k in [
        "not flowing","not reflecting",
        "not showing","missing",
        "not coming","transfer not reflected"
    ]):
        return "IT – System linkage issue"

    # LEARNED KEYWORDS
    for category, keywords in category_keywords.items():

        for kw in keywords:

            if kw in text:
                return category

    return None


# ---------------------------------------
# SUMMARY GENERATOR
# ---------------------------------------

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


# ---------------------------------------
# LOAD TRAINING DATA
# ---------------------------------------

train_df = pd.read_excel("issue category.xlsx")

train_df.columns = train_df.columns.str.strip()

train_df["combined_text"] = combine_columns(train_df).apply(clean_text)

train_df = train_df[train_df["combined_text"]!=""]

X_train = train_df["combined_text"]
y_train = train_df["Issue category"]


# learn keywords
category_keywords = learn_category_keywords(train_df)


# ---------------------------------------
# MACHINE LEARNING MODEL
# ---------------------------------------

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


# ---------------------------------------
# FILE UPLOAD
# ---------------------------------------

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
                confidence.append(0.99)

            else:
                pred = model.predict([text])[0]
                predictions.append(pred)
                confidence.append(0.95)

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
