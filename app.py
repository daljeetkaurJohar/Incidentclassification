import streamlit as st
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.title("AI Incident Category Classifier")

# Load training dataset
train_df = pd.read_excel("issue category.xlsx")

train_df = train_df.dropna(subset=["Ticket Description","Issue category"])

X_train = train_df["Ticket Description"].astype(str)
y_train = train_df["Issue category"]

# ML pipeline
model = Pipeline([
("tfidf", TfidfVectorizer(stop_words="english")),
("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

st.success("Training model ready")

uploaded_file = st.file_uploader("Upload Incident File", type=["xlsx"])

TEXT_COLUMNS = [
"Ticket Summary",
"Ticket Details",
"Ticket Description",
"Solution",
"Additional comments",
"Work notes",
"Remarks",
"Support required for issues' closure",
"Any reason for delay"
]

if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel, sheet_name=sheet)

        df.columns = df.columns.str.strip()

        available_cols = [c for c in TEXT_COLUMNS if c in df.columns]

        if len(available_cols) == 0:

            df["Predicted Category"] = "No text columns"
            df["Confidence"] = 0
            df["Rewritten Summary"] = ""

            df.to_excel(writer, sheet_name=sheet, index=False)
            continue

        # combine text fields
        df["combined_text"] = df[available_cols].fillna("").astype(str).agg(" ".join, axis=1)

        predictions = model.predict(df["combined_text"])

        probs = model.predict_proba(df["combined_text"])

        confidence = probs.max(axis=1)

        df["Predicted Category"] = predictions
        df["Confidence"] = confidence.round(3)

        # rewritten summary
        df["Rewritten Summary"] = df["combined_text"].apply(lambda x: " ".join(x.split()[:30]))

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
