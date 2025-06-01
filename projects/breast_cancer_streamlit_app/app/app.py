import streamlit as st
import pandas as pd
from joblib import dump, load

def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
if __name__ == '__main__':
    main()