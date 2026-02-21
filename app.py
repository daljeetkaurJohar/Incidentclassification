import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model")
    model = AutoModelForSequenceClassification.from_pretrained("model")
    label_encoder = joblib.load("label_encoder.pkl")
    model.eval()
    return tokenizer, model, label_encoder

tokenizer, model, label_encoder = load_model()

# -----------------------------
# Rule Dictionary
# -----------------------------
RULES = {
    "System linkage issue": [
        "not flowing", "not reflecting", "not appearing",
        "unable to pull", "integration", "not pushed",
        "version movement", "integrated version"
    ],
    "Mapping missing from user": [
        "not mapped", "mapping not done"
    ],
    "Multiple versions issue in excel": [
        "excel version", "difference in excel",
        "rate difference"
    ],
    "Masterdata - delayed input from user": [
        "rate missing", "material not visible",
        "bulk rate", "new scheme addition"
    ],
    "User knowledge gap": [
        "how to", "cannot see", "not visible"
    ]
}

CONFIDENCE_THRESHOLD = 0.65

# -----------------------------
# Rule Classification
# -----------------------------
def rule_based(text):
    text = text.lower()
    for category, keywords in RULES.items():
        for kw in keywords:
            if kw in text:
                return category, 0.95, "Rule"
    return None, None, None

# -----------------------------
# ML Classification
# -----------------------------
def ml_predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()
    label = label_encoder.inverse_transform([pred])[0]

    return label, confidence, "ML"

# -----------------------------
# Hybrid Classification
# -----------------------------
def classify(text):

    # Rule First
    rule_cat, rule_conf, rule_src = rule_based(text)
    if rule_cat:
        return rule_cat, rule_conf, rule_src

    # ML Fallback
    ml_cat, ml_conf, ml_src = ml_predict(text)

    if ml_conf >= CONFIDENCE_THRESHOLD:
        return ml_cat, round(ml_conf, 3), ml_src

    return "Needs Manual Review", round(ml_conf, 3), "Review"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Enterprise Ticket Classification System")

uploaded_file = st.file_uploader("Upload Incident Excel File", type=["xlsx"])

if uploaded_file:

    df = pd.read_excel(uploaded_file)

    predictions = []
    confidences = []
    sources = []

    for _, row in df.iterrows():

        text = " ".join([
            str(row.get("Issue", "")),
            str(row.get("Description", "")),
            str(row.get("Remarks", "")),
            str(row.get("Ticket Description", ""))
        ])

        cat, conf, src = classify(text)

        predictions.append(cat)
        confidences.append(conf)
        sources.append(src)

    df["Predicted Category"] = predictions
    df["Confidence"] = confidences
    df["Classification Source"] = sources

    st.success("Classification Complete")

    st.dataframe(df.head())

    output_file = "classified_output.xlsx"
    df.to_excel(output_file, index=False)

    with open(output_file, "rb") as f:
        st.download_button("Download Classified File", f, file_name=output_file)
