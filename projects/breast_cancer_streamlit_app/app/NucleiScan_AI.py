import streamlit as st

def main():
    st.set_page_config(
        page_title="NucleiScan AI",
        page_icon=":microscope:",
        layout="centered"
    )
    
    with st.sidebar:
        st.page_link("NucleiScan_AI.py", label="Home", icon="🏠")
        st.page_link("pages/Predictor.py", label="Predictor", icon="🔬")
        st.page_link("pages/Clinical_Assistant.py", label="Clinical Assistant", icon="💡")
    
    # Professional CSS styling
    st.markdown("""
    <style>
        :root {
            --primary: #4a6fa5;
            --secondary: #166088;
            --accent: #4fc3f7;
            --light: #f8f9fa;
            --dark: #212529;
            --success: #28a745;
        }
        
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: var(--secondary);
            text-align: center;
            margin-bottom: 1rem;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #166088 0%, #4a6fa5 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: var(--light);
            text-align: center;
            margin-bottom: 3rem;
            line-height: 1.6;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            opacity: 0.9;
        }
        
        .feature-section {
            background: #f8fafc;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        
        .feature-card {
            padding: 2rem 1.5rem;
            border-radius: 12px;
            background: white;
            box-shadow: 0 6px 15px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            height: 100%;
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 20px rgba(0,0,0,0.1);
            border-color: var(--accent);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1.2rem;
            color: var(--primary);
        }
        
        .feature-title {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--secondary);
        }
        
        .feature-desc {
            color: #5a6a7a;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }
        
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            border-radius: 8px;
            font-weight: 500;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            color: white;
            transition: all 0.3s;
            border: none;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        .footer {
            text-align: center;
            margin-top: 4rem;
            color: #95a5a6;
            font-size: 0.9rem;
            padding-top: 1.5rem;
            border-top: 1px solid #eee;
        }
        
        .logo-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1.5rem;
        }
        
        .logo {
            font-size: 3.5rem;
            color: var(--primary);
        }
        
        @media (max-width: 768px) {
            .main-title {
                font-size: 2.2rem;
            }
            .subtitle {
                font-size: 1rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section with Logo
    st.markdown('<div class="logo-container"><div class="logo">🔬</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">NucleiScan AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'Advanced AI-powered breast cancer diagnostics platform combining deep learning analysis '
        'with clinical decision support for pathologists and oncologists.'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Feature Cards Section
    st.markdown("### Core Features")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">NucleiScan Predictor</div>
            <div class="feature-desc">
                Advanced AI analysis of nuclear morphology features with 96.2% clinical accuracy 
                in malignancy risk assessment. Supports standard digital pathology formats.
            </div>
            <div class="feature-desc">
                <strong>Key Metrics:</strong> Nuclear pleomorphism, chromatin pattern, mitotic count
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Launch Nuclei Analysis", key="predictor", use_container_width=True):
            st.switch_page("pages/Predictor.py")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Clinical Decision Assistant</div>
            <div class="feature-desc">
                Evidence-based clinical guidance powered by the latest NCCN guidelines and 
                peer-reviewed research. Get second opinions on complex cases.
            </div>
            <div class="feature-desc">
                <strong>Features:</strong> Treatment recommendations, diagnostic criteria, prognostic factors
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Consult Clinical Assistant", key="chatbot", use_container_width=True):
            st.switch_page("pages/Clinical_Assistant.py")
    
    # How the Project Works Section
    st.markdown("---")
    st.markdown("### How NucleiScan AI Works")

    cols = st.columns(5)
    with cols[0]:
        st.markdown("**1. Upload Images**")
        st.markdown("""
        <div style='color: #5a6a7a; font-size: 0.9rem; line-height: 1.5;'>
            Adjust nuclear characteristics via the sidebar controls including:
            radius, texture, concavity, and more.
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown("**2. AI Analysis**")
        st.markdown("""
        <div style='color: #5a6a7a; font-size: 0.9rem; line-height: 1.5;'>
        Deep learning analyzes nuclear features with 96.2% accuracy. 
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown("**3. Generate Report**")
        st.markdown("""
        <div style='color: #5a6a7a; font-size: 0.9rem; line-height: 1.5;'>
        System creates reports with risk scores. Real-time visualization of features in an interactive radar chart
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown("**4. Clinical Decision**")
        st.markdown("""
        <div style='color: #5a6a7a; font-size: 0.9rem; line-height: 1.5;'>
        Instant prediction with confidence scoring:
        <br><br>
        • Malignant/Benign classification<br>
        • Risk probability percentage<br>
        • Clinical decision support
        </div>
        """, unsafe_allow_html=True)
        
    with cols[4]:
        st.markdown("**5. Clinical Assistant Chatbot**")
        st.markdown("""
        <div style='color: #5a6a7a; font-size: 0.9rem; line-height: 1.5;'>
            Get expert insights on-demand:<br>
            • Ask about diagnostic criteria<br>
            • Request treatment options<br>
            • Clarify pathology findings<br>
            • Get second opinions
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div>NucleiScan AI by IHEANYI, FAVOUR CHISOM</div>
        <div style="margin-top: 0.5rem; font-size: 0.8rem;">
            For clinical use only | Not a substitute for professional medical judgment
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()