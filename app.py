import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Incident Category Classifier")

# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load training dataset
train_df = pd.read_excel("issue category.xlsx")

train_text = train_df["Ticket Description"].astype(str)
train_labels = train_df["Issue category"]

train_embeddings = model.encode(train_text.tolist())

st.success("Training dataset loaded")

uploaded_file = st.file_uploader("Upload Incident File", type=["xlsx"])

# Possible text columns in sheets
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

# Rule based categorization (high precision)
RULES = {
    "IT – Master Data/ mapping issue": ["mapping", "master data", "mdm"],
    "IT - System Access issue": ["login", "access", "permission", "authorization"],
    "IT - System linkage issue": ["interface", "integration", "sync", "link"],
    "IT – System Version issue": ["version", "upgrade", "patch"],
    "User - Logic mistakes in excel vs system": ["excel", "formula", "calculation"],
    "User - Multiple versions issue in excel": ["multiple file", "duplicate excel"],
}

def rule_based_category(text):

    text = text.lower()

    for category, keywords in RULES.items():
        for k in keywords:
            if k in text:
                return category

    return None


if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel, sheet_name=sheet)

        available_cols = [c for c in TEXT_COLUMNS if c in df.columns]

        if len(available_cols) == 0:

            df["Predicted Category"] = "No text columns found"
            df.to_excel(writer, sheet_name=sheet, index=False)
            continue

        # Combine useful columns
        df["combined_text"] = df[available_cols].astype(str).agg(" ".join, axis=1)

        predictions = []
        confidence = []

        embeddings = model.encode(df["combined_text"].tolist())

        for i, emb in enumerate(embeddings):

            text = df["combined_text"].iloc[i]

            # 1. Rule based classification first
            rule_cat = rule_based_category(text)

            if rule_cat:
                predictions.append(rule_cat)
                confidence.append(0.95)
                continue

            # 2. AI similarity fallback
            sim = cosine_similarity([emb], train_embeddings)[0]

            idx = sim.argmax()

            predictions.append(train_labels.iloc[idx])
            confidence.append(round(float(sim[idx]),3))

        df["Predicted Category"] = predictions
        df["Confidence"] = confidence

        # Add Updated Summary column
        df["Updated Summary"] = df["combined_text"].str.slice(0,200)

        df.drop(columns=["combined_text"], inplace=True)

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    st.success("Categorization completed")

    st.download_button(
        "Download Categorized File",
        data=output.getvalue(),
        file_name="categorized_tickets.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
