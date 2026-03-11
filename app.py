import streamlit as st
import pandas as pd
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Configuration ---
# This must match the filename in your folder exactly
REFERENCE_FILENAME = "issue category.xlsx - Sheet1.csv"

def train_classifier(mapping_df):
    """Trains the NLP model based on 'Issue category' and 'Ticket Description'."""
    # Clean data: remove rows missing essential info
    df_clean = mapping_df.dropna(subset=['Ticket Description', 'Issue category'])
    
    X = df_clean['Ticket Description']
    y = df_clean['Issue category']
    
    # Create a text processing pipeline
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X, y)
    return model

def generate_clean_summary(text):
    """Summarizes long descriptions into a short 'Update' string."""
    if pd.isna(text) or text == "": 
        return "No description"
    # Extract the core action (text before 'like', '-', or ',')
    core = text.split("like")[0].split("-")[0].split(",")[0].strip()
    return core[:75] + "..." if len(core) > 75 else core

# --- Streamlit UI ---
st.set_page_config(page_title="ABP Issue Processor", layout="wide")
st.title("📑 ABP FY'27 Issue Categorizer & Summarizer")

# 1. Reference Data Loading Logic
model = None
st.sidebar.header("Reference Mapping Settings")

# Check if file exists in the current directory
if os.path.exists(REFERENCE_FILENAME):
    try:
        ref_df = pd.read_csv(REFERENCE_FILENAME)
        model = train_classifier(ref_df)
        st.sidebar.success(f"✅ Auto-loaded: {REFERENCE_FILENAME}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
else:
    # Backup: Manual upload if the file isn't found in the directory
    st.sidebar.warning(f"⚠️ '{REFERENCE_FILENAME}' not found in script folder.")
    uploaded_ref = st.sidebar.file_uploader("Upload 'issue category.xlsx - Sheet1.csv' manually", type="csv")
    if uploaded_ref:
        ref_df = pd.read_csv(uploaded_ref)
        model = train_classifier(ref_df)
        st.sidebar.success("✅ Manual Mapping Loaded")

# 2. Main Processing Logic
if model:
    st.info("Upload your new issue list (CSV or XLSX) to categorize them based on the mapping.")
    uploaded_file = st.file_uploader("Upload New Issues", type=["csv", "xlsx"])

    if uploaded_file:
        # Load input file
        if uploaded_file.name.endswith('.csv'):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        # Check for required column
        if 'Ticket Description' in input_df.columns:
            with st.spinner('Categorizing and generating summaries...'):
                # Apply Model
                input_df['Categorized Issue'] = model.predict(input_df['Ticket Description'].fillna(""))
                # Apply Summarization
                input_df['Update Summary'] = input_df['Ticket Description'].apply(generate_clean_summary)

            st.subheader("Preview of Results")
            st.dataframe(input_df[['Ticket Description', 'Categorized Issue', 'Update Summary']].head(10))

            # 3. Export to Excel (XLSX)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                input_df.to_excel(writer, index=False, sheet_name='ABP_Categorized_Data')
            
            st.divider()
            st.download_button(
                label="📥 Download Updated XLSX",
                data=output.getvalue(),
                file_name="ABP_Categorized_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("The uploaded file must contain a 'Ticket Description' column.")
else:
    st.warning("Please place the mapping file in the folder or upload it via the sidebar to enable categorization.")
