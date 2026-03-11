import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Ticket Categorization")

# Load AI model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load training data from repo
train_df = pd.read_excel("issue_category.xlsx")

X = train_df["Ticket Description"].astype(str)
y = train_df["Issue category"]

train_embeddings = model.encode(X.tolist())

st.success("Training data loaded")

# User uploads new file
uploaded_file = st.file_uploader("Upload new ticket file", type=["xlsx"])

if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel, sheet_name=sheet)

        text = df["Ticket Description"].astype(str)

        new_embeddings = model.encode(text.tolist())

        predictions = []

        for emb in new_embeddings:

            sim = cosine_similarity([emb], train_embeddings)

            idx = sim.argmax()

            predictions.append(y.iloc[idx])

        df["Predicted Category"] = predictions

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    st.success("Categorization Completed")

    st.download_button(
        "Download Categorized File",
        data=output.getvalue(),
        file_name="categorized_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
