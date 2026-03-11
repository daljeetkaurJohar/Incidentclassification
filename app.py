import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Customer Support Ticket Analyzer")

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Ticket Categories
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

# Encode categories
category_embeddings = model.encode(categories)


def categorize_ticket(text):

    text = str(text)

    ticket_embedding = model.encode([text])

    similarity = cosine_similarity(ticket_embedding, category_embeddings)

    index = similarity.argmax()

    return categories[index]


def rewrite_summary(row):

    text = " ".join([str(v) for v in row if pd.notna(v)])

    text = text.replace("\n"," ")

    if len(text) > 200:
        text = text[:200] + "..."

    return text


uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:

    excel = pd.ExcelFile(uploaded_file)

    output = BytesIO()

    writer = pd.ExcelWriter(output, engine="openpyxl")

    for sheet in excel.sheet_names:

        df = pd.read_excel(excel, sheet_name=sheet)

        text_cols = df.select_dtypes(include="object").columns

        df["combined_text"] = df[text_cols].astype(str).agg(" ".join, axis=1)

        df["Category"] = df["combined_text"].apply(categorize_ticket)

        if "Ticket Summary" in df.columns:

            df["Ticket Summary"] = df.apply(rewrite_summary, axis=1)

        df.drop(columns=["combined_text"], inplace=True)

        df.to_excel(writer, sheet_name=sheet, index=False)

    writer.close()

    st.success("Processing completed")

    st.download_button(
        label="Download Updated Excel",
        data=output.getvalue(),
        file_name="Updated_Tickets.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
