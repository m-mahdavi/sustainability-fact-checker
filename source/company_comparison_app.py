import os
import sys
import pandas as pd
import streamlit as st
import PIL

sys.path.append("../source")
import company


st.set_page_config(page_title="Company Comparison", layout="wide")
st.title("📊 Company Comparison")

available_companies = [
    folder for folder in os.listdir("../documents")
    if os.path.isdir(os.path.join("../documents", folder))
]

col1, col2 = st.columns(2)
with col1:
    company1_name = st.selectbox("Select Company 1", available_companies, key="c1")
with col2:
    company2_name = st.selectbox("Select Company 2", available_companies, key="c2")

selected_objective = st.text_input("Enter a Sustainability Objective to Compare")

def get_or_load_company(name, key):
    if key not in st.session_state or st.session_state[key + "_name"] != name:
        st.session_state[key + "_name"] = name
        comp = company.Company(name)
        comp.load_database()
        st.session_state[key] = comp
    return st.session_state[key]

if selected_objective and st.button("🔍 Compare Companies"):
    with st.spinner("Verifying objective for both companies..."):
        c1 = get_or_load_company(company1_name, "company1")
        c2 = get_or_load_company(company2_name, "company2")

        text_ev_1, img_ev_1 = c1.retrieve_evidence(selected_objective)
        report_1 = c1.verify_objective(selected_objective, text_ev_1, llm_model="llama3.2")

        text_ev_2, img_ev_2 = c2.retrieve_evidence(selected_objective)
        report_2 = c2.verify_objective(selected_objective, text_ev_2, llm_model="llama3.2")

    st.markdown("## 📝 Verification Reports")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {company1_name}")
        st.markdown(report_1)
    with col2:
        st.markdown(f"### {company2_name}")
        st.markdown(report_2)

    st.markdown("## 📸 Image Evidence")
    img_col1, img_col2 = st.columns(2)

    def show_images(images, container):
        if images:
            with container:
                cols = st.columns(3)
                for i, image in enumerate(images[:3]):
                    path = image["record"]["image_path"]
                    if os.path.exists(path):
                        with cols[i % 3]:
                            st.image(PIL.Image.open(path), width=150)
                    else:
                        st.warning(f"Image not found: {path}")

    show_images(img_ev_1, img_col1)
    show_images(img_ev_2, img_col2)
