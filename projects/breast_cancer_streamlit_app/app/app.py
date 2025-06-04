import streamlit as st
import pandas as pd
import numpy as np
from joblib import dump, load
import plotly.graph_objects as go

# Importing models and libraries for the chatbot
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
import os

def get_clean_data():
    data = pd.read_csv("../data/data.csv")    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)    
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
        ]
    input_dict = {}
    
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict  

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    return scaled_dict

def get_radar_chart(input_data):
    
    input_data = get_scaled_values(input_data)
    
    categories = ['Radius', 'Texture', 'Perimeter',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
          theta=categories,
          fill='toself',
          name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']
        ],
          theta=categories,
          fill='toself',
          name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
          theta=categories,
          fill='toself',
          name='Worst Value'
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=True
    )
    
    return fig


def add_predictions(input_data):
    model = load('../models/logistic_regression_model.joblib')
    scaler = load('../models/scaler.joblib')
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is: ")
    
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])

def setup_medical_chatbot():
    """Initialize the medical chatbot components"""
    # Setup Hugging Face LLM (BioMistral-7B)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "*******"
    
    llm = HuggingFaceHub(
        repo_id="BioMistral/BioMistral-7B",
        model_kwargs={"temperature": 0.5, "max_length": 500}
    )
    
    # Create simple medical knowledge base
    medical_knowledge = """
    Breast cancer diagnostic criteria:
    - Malignant nuclei: Irregular borders, high nuclear-cytoplasmic ratio
    - Key metrics: Radius > 15μm, Concavity > 0.08, Fractal Dimension < 0.065
    Common terms:
    - DCIS: Ductal carcinoma in situ (non-invasive)
    - IDC: Invasive ductal carcinoma
    """
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return llm, memory, medical_knowledge

def medical_chat(question, llm, memory, context=""):
    """Generate medical responses with safety checks"""
    if any(word in question.lower() for word in ["emergency", "pain", "urgent"]):
        return "Please contact your healthcare provider immediately for urgent concerns."
    
    prompt = f"""As a cytology specialist, answer concisely (2-3 sentences max):
    Context: {context}
    Question: {question}
    Answer (professional tone):"""
    
    response = llm(prompt)
    return response.strip()
        
def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("../assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    # Initialize chatbot (only once)
    if "chatbot" not in st.session_state:
        st.session_state.llm, st.session_state.memory, st.session_state.medical_kb = setup_medical_chatbot()
    
    with st.container():
        st.title("NucleiScan AI: Breast Cancer Predictor")
        st.write("NucleiScan AI analyzes 30 cell nuclei characteristics from breast tissue samples to predict malignancy risk. This clinical decision-support tool evaluates mean, standard error, and worst-case measurements (including radius, concavity, and texture) to assist pathologists in rapid assessment. Features interactive sliders for manual input or lab system integration.")
        
    col1, col2 = st.columns([4,1])
    
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        
        with st.container():
            st.markdown("*<span class='notice important'>Important</span>: This app assists medical professionals in making a diagnosis, but should not be used as a substitute for professional diagnosis.*", unsafe_allow_html=True)

    with col2:
        # ---CHATBOT SECTION -----
        st.markdown("### Medical Query Assistant")
        st.caption("Ask about your results or general breast cancer questions")
        
        # Display chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for msg in st.session_state.messages:
            st.chat_messages(msg["role"]).write(msg["content"])
            
        # Chat input
        if prompt := st.chat_input("Type your question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Consulting NucleiScan AI..."):
                
                # Generate response with current metrics as context
                current_metrics = f"Radius: {input_data['radius_mean']:.2f}, Concavity: {input_data['concavity_mean']:.2f}"
                
                response = medical_chat(
                    prompt,
                    st.session_state.llm,
                    st.session_state.memory,
                    context=current_metrics
                )
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
               
        add_predictions(input_data)
        
        
        
if __name__ == '__main__':
    main()