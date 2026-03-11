import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Customer Support Ticket Analyzer")

# Load AI embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Categories
categories = [
"IT - System linkage issue",
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

# Precompute category embeddings
category_embeddings = model.encode(categories)


def categorize_ticket(text):

    ticket_embedding = model.encode([text])

    similarity = cosine_similarity(ticket_embedding, category_embeddings)

    index = similarity.argmax()

    return categories[index]


def rewrite_summary(text):

    text = str(text)

    # Simple AI-like cleaning
    text = text.replace("\n"," ")
    text = text.strip()

    if len(text) > 180:
        text = text[:180] + "..."

    return text


uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel, sheet_name=sheet)

        text_columns = df.select_dtypes(include="object").columns

        df["combined_text"] = df[text_columns].astype(str).agg(" ".join, axis=1)

        # Categorize tickets
        df["Category"] = df["combined_text"].apply(categorize_ticket)

        # Rewrite ticket summary
        if "Ticket Summary" in df.columns:
            df["Ticket Summary"] = df["combined_text"].apply(rewrite_summary)

        df.drop(columns=["combined_text"], inplace=True)

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    st.success("AI Processing Completed")

    st.download_button(
        label="Download Updated Excel",
        data=output.getvalue(),
        file_name="AI_Ticket_Analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
