import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Incident Category Classifier")

# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load training data
train_df = pd.read_excel("issue category.xlsx")

train_text = train_df["Ticket Description"].astype(str)
train_labels = train_df["Issue category"]

train_embeddings = model.encode(train_text.tolist())

st.success("Training dataset loaded")

uploaded_file = st.file_uploader("Upload Incident File", type=["xlsx"])

# columns that may contain useful text
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

        # find which text columns exist in this sheet
        available_cols = [c for c in TEXT_COLUMNS if c in df.columns]

        if len(available_cols) == 0:
            df["Predicted Category"] = "No text columns found"
            df.to_excel(writer, sheet_name=sheet, index=False)
            continue

        # combine all text columns
        df["combined_text"] = df[available_cols].astype(str).agg(" ".join, axis=1)

        new_embeddings = model.encode(df["combined_text"].tolist())

        predictions = []
        confidence = []

        for emb in new_embeddings:

            sim = cosine_similarity([emb], train_embeddings)[0]

            idx = sim.argmax()

            predictions.append(train_labels.iloc[idx])
            confidence.append(round(float(sim[idx]),3))

        df["Predicted Category"] = predictions
        df["Confidence"] = confidence

        df.drop(columns=["combined_text"], inplace=True)

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    st.success("Categorization completed")

    st.download_button(
        "Download Categorized File",
        data=output.getvalue(),
        file_name="categorized_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
