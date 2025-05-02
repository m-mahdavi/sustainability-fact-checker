# Sustainability Fact Checker

**Sustainability Fact Checker** is a system designed to verify corporate sustainability claims using multimodal evidence from textual and visual data. It supports auditors, investors, and other stakeholders by enhancing the transparency and verifiability of sustainability reporting.

## Features

- 🧠 Extracts information from sustainability reports (text + images)
- 🔍 Uses CLIP and other models to retrieve supporting visual and textual evidence
- 📊 Generates fact-checking reports
- 💡 Provides a Streamlit interface for interactive exploration

## Installation

```bash
git clone https://github.com/m-mahdavi/sustainability-fact-checker.git
cd sustainability-fact-checker
pip install .
```

### Usage 

- To use the system in a Jupyter Notebook interface, run: ```notebooks/fact_checking.ipynb```
- To launch the web-based user interface with Streamlit, run: ```streamlit run source/fact_checking_app.py```
