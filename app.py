import streamlit as st
import pandas as pd
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Ticket Categorization System")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Upload training file
training_file = st.file_uploader("Upload training file (categorized tickets)", type=["xlsx"])

# Upload new file
new_file = st.file_uploader("Upload new tickets file", type=["xlsx"])

if training_file:

    train_df = pd.read_excel(training_file)

    X = train_df["Ticket Description"].astype(str)
    y = train_df["Issue category"]

    embeddings = model.encode(X.tolist())

    st.success("Training data loaded")

    if new_file:

        new_excel = pd.ExcelFile(new_file)

        output = BytesIO()
        writer = pd.ExcelWriter(output, engine="openpyxl")

        for sheet in new_excel.sheet_names:

            df = pd.read_excel(new_excel, sheet_name=sheet)

            text = df["Ticket Description"].astype(str)

            new_embeddings = model.encode(text.tolist())

            predictions = []

            for emb in new_embeddings:

                sim = cosine_similarity([emb], embeddings)

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
