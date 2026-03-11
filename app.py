import streamlit as st
import pandas as pd
import re
from io import BytesIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.title("Incident Category Classifier")

# ---------------------------
# VALID CATEGORIES
# ---------------------------

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


# ---------------------------
# KEYWORD RULE ENGINE
# ---------------------------

RULES = {
"IT - System Access issue":[
"login","access","permission","authorization"
],

"IT – Master Data/ mapping issue":[
"mapping","master data mapping","mdm mapping"
],

"User - Mapping missing":[
"mapping missing","no mapping"
],

"User – Master data delayed input":[
"master data delay","delayed master data"
],

"User - Logic mistakes in excel vs system":[
"excel mismatch","formula issue"
],

"User - Multiple versions issue in excel":[
"multiple excel","duplicate file"
],

"IT – System Version issue":[
"version issue","upgrade problem"
],

"User - Logic changes during ABP":[
"logic change","abp logic"
],

"IT – Data entry handholding":[
"how to enter","data entry help"
],

"User – Master data incorporation in system":[
"add master data","incorporate master data"
]
}


# ---------------------------
# TEXT CLEANING
# ---------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'\n',' ',text)
    text = re.sub(r'\s+',' ',text)

    return text.strip()


# ---------------------------
# COMBINE TEXT FIELDS
# ---------------------------

TEXT_COLUMNS = [
"Ticket Summary",
"Ticket Details",
"Ticket Description",
"Work notes",
"Additional comments",
"Remarks",
"Solution"
]

def combine_columns(df):

    available = [c for c in TEXT_COLUMNS if c in df.columns]

    if not available:
        return pd.Series([""]*len(df))

    return df[available].fillna("").astype(str).agg(" ".join,axis=1)


# ---------------------------
# RULE-BASED CLASSIFICATION
# ---------------------------

def rule_classifier(text):

    for cat,keywords in RULES.items():

        for k in keywords:

            if k in text:

                return cat

    return None


# ---------------------------
# TRAIN ML MODEL
# ---------------------------

train_df = pd.read_excel("issue category.xlsx")

train_df.columns = train_df.columns.str.strip()

train_df["combined_text"] = combine_columns(train_df).apply(clean_text)

train_df = train_df[train_df["combined_text"]!=""]

X_train = train_df["combined_text"]
y_train = train_df["Issue category"]

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


# ---------------------------
# FILE UPLOAD
# ---------------------------

uploaded_file = st.file_uploader("Upload Incident Excel File", type=["xlsx"])


if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()

    writer = pd.ExcelWriter(output, engine="openpyxl")


    for sheet in excel.sheet_names:

        df = pd.read_excel(excel,sheet_name=sheet)

        df.columns = df.columns.str.strip()

        df["combined_text"] = combine_columns(df).apply(clean_text)

        predictions=[]
        confidence=[]


        for text in df["combined_text"]:

            # RULE FIRST
            rule = rule_classifier(text)

            if rule:
                predictions.append(rule)
                confidence.append(0.97)
                continue

            # ML FALLBACK
            pred = model.predict([text])[0]

            prob = model.predict_proba([text]).max()

            predictions.append(pred)
            confidence.append(round(prob,3))


        df["Predicted Category"]=predictions
        df["Confidence"]=confidence


        # REWRITTEN SUMMARY
        df["Rewritten Summary"]=df["combined_text"].apply(
        lambda x:" ".join(x.split()[:20]).capitalize()
        )


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
