import os
import io
import pickle
import random
import pandas as pd
import PIL.Image
import streamlit as st


OUTPUTS_DIR = "../outputs"
RESULTS_FILE = "../outputs/human_evaluation_results.csv"
FIXED_IMG_WIDTH = 400  

st.set_page_config(
    page_title="Pairwise Human Evaluation",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_pickle_files(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")]
    files.sort()
    return files

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_result(filename, evaluator, preference):
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
    else:
        df = pd.DataFrame(columns=["pickle_file", "evaluator", "preference"])
    df = pd.concat(
        [df, pd.DataFrame([[filename, evaluator, preference]],
                          columns=["pickle_file", "evaluator", "preference"])],
        ignore_index=True
    )
    df.to_csv(RESULTS_FILE, index=False)

def bytes_to_image(image_bytes):
    if image_bytes is None:
        return None
    return PIL.Image.open(io.BytesIO(image_bytes))


st.title("Pairwise Human Evaluation for Sustainability Objective Fact-Checking")

evaluator = st.text_input("Enter your name:")

if evaluator:
    pickle_files = load_pickle_files(OUTPUTS_DIR)

    if "file_index" not in st.session_state:
        st.session_state.file_index = 0

    if "pair_order" not in st.session_state:
        st.session_state.pair_order = None

    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    if st.session_state.file_index < len(pickle_files):
        file_path = pickle_files[st.session_state.file_index]
        output = load_pickle(file_path)

        if st.session_state.last_file != file_path or st.session_state.pair_order is None:
            items = [
                ("System", output["system_verification_report"], output["system_image_bytes"]),
                ("Baseline", output["baseline_verification_report"], output["baseline_image_bytes"])
            ]
            random.shuffle(items)
            st.session_state.pair_order = items
            st.session_state.last_file = file_path

        items = st.session_state.pair_order

        st.subheader(f"Objective: {output['objective']}")
        st.caption(f"File {st.session_state.file_index + 1} of {len(pickle_files)} — {os.path.basename(file_path)}")

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Option A")
            st.markdown(items[0][1]) 
            img_a = bytes_to_image(items[0][2])
            if img_a is not None:
                st.image(img_a, width=FIXED_IMG_WIDTH)
            else:
                st.info("No image for Option A.")
            if st.button("Choose Option A"):
                save_result(os.path.basename(file_path), evaluator, items[0][0])
                st.session_state.file_index += 1
                st.session_state.pair_order = None
                st.rerun()
        
        with col2:
            st.markdown("### Option B")
            st.markdown(items[1][1])  # render markdown
            img_b = bytes_to_image(items[1][2])
            if img_b is not None:
                st.image(img_b, width=FIXED_IMG_WIDTH)  # consistent size
            else:
                st.info("No image for Option B.")
            if st.button("Choose Option B"):
                save_result(os.path.basename(file_path), evaluator, items[1][0])
                st.session_state.file_index += 1
                st.session_state.pair_order = None
                st.rerun()

        st.markdown("---")
        if st.button("Skip this example 🚫"):
            save_result(os.path.basename(file_path), evaluator, "Skip")
            st.session_state.file_index += 1
            st.session_state.pair_order = None
            st.rerun()

    else:
        st.success("✅ You have completed all evaluations. Thank you!")

else:
    st.info("Please enter your name to begin.")
