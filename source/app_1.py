import os
import sys
import pandas as pd
import streamlit as st
import PIL

sys.path.append("../source")
import company

st.set_page_config(page_title="Sustainability Objective Verification", layout="wide")
st.title("📊 Sustainability Objective Verification")

available_companies = [
    folder for folder in os.listdir("../documents")
    if os.path.isdir(os.path.join("../documents", folder))
]

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    company_name = st.selectbox("Select a Company", available_companies)

if "current_company" not in st.session_state or st.session_state.current_company != company_name:
    st.session_state.current_company = company_name
    st.session_state.company_object = company.Company(company_name)
    st.session_state.company_object.load_database()

c = st.session_state.company_object

objectives_dir = os.path.join("../objectives", company_name)
objective_files = [
    file for file in os.listdir(objectives_dir)
    if file.endswith(".csv")
]

with col2:
    objectives_file = st.selectbox("Select a Report", objective_files)

objectives_path = os.path.join(objectives_dir, objectives_file)
df = pd.read_csv(objectives_path)
objectives = df["Text Blocks"].dropna().unique()

with col3:
    selected_objective = st.selectbox("Select an Objective", options=objectives)

if st.button("🕵️‍♂️ Verify Objective"):
    with st.spinner("🔍 Verifying the Objective..."):
        text_evidence, image_evidence = c.retrieve_evidence(selected_objective)
        verification_report = c.verify_objective(
            objective=selected_objective,
            text_evidence=text_evidence,
            llm_model="llama3.2"
        )

    st.markdown("### 📝 Verification Report")
    st.markdown(verification_report)

    if image_evidence:
        st.markdown("### 📸 Image Evidence")
        img_cols = st.columns(3)
        for i, image in enumerate(image_evidence[:3]):
            image_path = image["record"]["image_path"]
            with img_cols[i]:
                if os.path.exists(image_path):
                    st.image(PIL.Image.open(image_path), width=150)
                else:
                    st.warning(f"Image not found: {image_path}")
