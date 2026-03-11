import pandas as pd
import streamlit as st
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Helper Functions ---
def train_model(mapping_df):
    """Trains the NLP model to recognize 'Issue category'."""
    X = mapping_df['Ticket Description'].fillna('')
    y = mapping_df['Issue category'].fillna('Uncategorized')
    model = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())
    model.fit(X, y)
    return model

def generate_summary(text):
    """Creates a brief summary by extracting the first relevant phrase."""
    if pd.isna(text): return ""
    # Simple logic: extract the core action before the first hyphen or long list
    summary = text.split('-')[0].split(',')[0]
    return summary[:60] + "..." if len(summary) > 60 else summary

# --- Streamlit UI ---
st.set_page_config(page_title="ABP Issue Processor", layout="wide")
st.title("📊 ABP Issue Processor & Summarizer")

# Load the reference mapping (your provided CSV)
@st.cache_data
def load_reference():
    # Ensure your sample file is named 'issue_category_sample.csv'
    return pd.read_csv("issue category.xlsx")

try:
    reference_df = load_reference()
    model = train_model(reference_df)
    st.sidebar.success("✅ Reference Model Loaded")
except Exception:
    st.sidebar.error("⚠️ 'issue category.xlsx' not found.")

# File Uploader
uploaded_file = st.file_uploader("Upload New Issues File (CSV or XLSX)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'Ticket Description' in df.columns:
        with st.spinner('Categorizing and Summarizing...'):
            # 1. Predict Category
            df['Predicted Category'] = model.predict(df['Ticket Description'].fillna(''))
            
            # 2. Generate Summary Update
            df['Update Summary'] = df['Ticket Description'].apply(generate_summary)

        st.subheader("Processed Data Preview")
        st.dataframe(df[['Ticket Description', 'Predicted Category', 'Update Summary']].head(10))

        # 3. Excel Download Option
        # We use a buffer to allow downloading without saving a file to disk
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Categorized Issues')
        
        st.divider()
        st.download_button(
            label="📥 Download Updated XLSX",
            data=buffer.getvalue(),
            file_name="ABP_Categorized_Update.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Error: The file must contain a 'Ticket Description' column.")
