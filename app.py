import streamlit as st
import pandas as pd
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Configuration ---
# Updated to match your exact Excel filename
REFERENCE_FILENAME = "issue category.xlsx" 

def train_classifier(mapping_df):
    """Trains the model based on 'Issue category' and 'Ticket Description'."""
    # Ensure we use the correct column names from your Excel sheet
    df_clean = mapping_df.dropna(subset=['Ticket Description', 'Issue category'])
    X = df_clean['Ticket Description']
    y = df_clean['Issue category']
    
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X, y)
    return model

def generate_clean_summary(text):
    """Summarizes the description for the 'Update' column."""
    if pd.isna(text) or text == "": 
        return "No description"
    # Extracts the main issue before details start
    core = str(text).split("like")[0].split("-")[0].split(",")[0].strip()
    return core[:75] + "..." if len(core) > 75 else core

# --- Streamlit UI ---
st.set_page_config(page_title="ABP Issue Processor", layout="wide")
st.title("📊 ABP FY'27 Excel Categorizer")

model = None
st.sidebar.header("Excel Mapping Setup")

# 1. Load Reference Excel
if os.path.exists(REFERENCE_FILENAME):
    try:
        # Reading 'Sheet1' specifically as seen in your file list
        ref_df = pd.read_excel(REFERENCE_FILENAME, sheet_name='Sheet1')
        model = train_classifier(ref_df)
        st.sidebar.success(f"✅ Auto-loaded: {REFERENCE_FILENAME}")
    except Exception as e:
        st.sidebar.error(f"Error reading Excel: {e}")
else:
    st.sidebar.warning(f"⚠️ '{REFERENCE_FILENAME}' not found.")
    uploaded_ref = st.sidebar.file_uploader("Upload Mapping Excel", type="xlsx")
    if uploaded_ref:
        ref_df = pd.read_excel(uploaded_ref)
        model = train_classifier(ref_df)
        st.sidebar.success("✅ Mapping Loaded")

# 2. Process New Files
if model:
    st.markdown("### Upload new issues to categorize and summarize")
    new_file = st.file_uploader("Choose an Excel file", type="xlsx")

    if new_file:
        input_df = pd.read_excel(new_file)

        if 'Ticket Description' in input_df.columns:
            with st.spinner('Processing...'):
                # Categorization & Summarization
                input_df['Categorized Issue'] = model.predict(input_df['Ticket Description'].fillna(""))
                input_df['Update Summary'] = input_df['Ticket Description'].apply(generate_clean_summary)

            st.subheader("Preview")
            st.dataframe(input_df[['Ticket Description', 'Categorized Issue', 'Update Summary']].head(10))

            # 3. Create Downloadable Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                input_df.to_excel(writer, index=False, sheet_name='Updated_Issues')
            
            st.divider()
            st.download_button(
                label="📥 Download Updated XLSX",
                data=output.getvalue(),
                file_name="ABP_Categorized_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Column 'Ticket Description' not found in this file.")
