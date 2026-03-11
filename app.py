import streamlit as st
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Core Logic ---

def generate_clean_summary(text):
    """Summarizes long descriptions into a short 'Update' string."""
    if pd.isna(text) or text == "":
        return "No description provided"
    
    # Extract the core action before the long list of parameters/materials
    # e.g., "Helping users to update data like production numbers..." -> "Helping users to update data"
    core = text.split("like")[0].split("-")[0].split(",")[0].strip()
    
    # Cap length for readability in Excel
    return core[:75] + "..." if len(core) > 75 else core

def train_classifier(mapping_df):
    """Trains a model based on your provided Issue Category sheet."""
    # Clean data: drop rows missing description or category
    df_clean = mapping_df.dropna(subset=['Ticket Description', 'Issue category'])
    
    X = df_clean['Ticket Description']
    y = df_clean['Issue category']
    
    # Create an NLP pipeline
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X, y)
    return model

# --- Streamlit Interface ---

st.set_page_config(page_title="ABP Issue Categorizer", layout="wide")
st.title("📑 ABP FY'27 Issue Categorizer & Summarizer")

# 1. Load Reference Data (Your mapping file)
@st.cache_data
def load_reference():
    # Replace with the actual filename of your mapping CSV
    return pd.read_csv("issue category.xlsx - Sheet1.csv")

try:
    ref_df = load_reference()
    model = train_classifier(ref_df)
    st.sidebar.success("✅ Categorization Model Active")
except Exception as e:
    st.sidebar.error("⚠️ Mapping file not found. Please ensure 'issue category.xlsx - Sheet1.csv' is in the folder.")

# 2. File Upload for New Issues
uploaded_file = st.file_uploader("Upload New Issue List", type=["csv", "xlsx"])

if uploaded_file:
    # Read input
    if uploaded_file.name.endswith('.csv'):
        input_df = pd.read_csv(uploaded_file)
    else:
        input_df = pd.read_excel(uploaded_file)

    if 'Ticket Description' in input_df.columns:
        with st.spinner('Processing categories and summaries...'):
            # Predict Category using the trained model
            input_df['Categorized Issue'] = model.predict(input_df['Ticket Description'].fillna(""))
            
            # Generate the summary update
            input_df['Update Summary'] = input_df['Ticket Description'].apply(generate_clean_summary)

        st.subheader("Preview of Categorized Data")
        st.dataframe(input_df[['Ticket Description', 'Categorized Issue', 'Update Summary']].head(10))

        # 3. Export to Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            input_df.to_excel(writer, index=False, sheet_name='Update_Categorized')
        
        st.divider()
        st.download_button(
            label="📥 Download Updated XLSX",
            data=output.getvalue(),
            file_name="ABP_Categorized_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Error: Uploaded file must have a 'Ticket Description' column.")
