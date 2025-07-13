import streamlit as st
import streamlit.components.v1 as components
import os

# Import page renderers
from pages import home, feature_matching, homography, image_retrieval, epipolar_geometry, innovation

# --- App Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Computer Vision Explorer",
    page_icon="🔍",
    initial_sidebar_state="collapsed"
)

# --- Session State Initialization ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = "dark"
    
if 'current_tab' not in st.session_state:
    st.session_state['current_tab'] = 0

# --- Theme Toggle Function ---
def toggle_theme():
    st.session_state['theme'] = "dark" if st.session_state['theme'] == "light" else "light"
    st.rerun()

# --- Tab Navigation Functions ---
def set_tab(tab_index):
    st.session_state['current_tab'] = tab_index
    st.rerun()

# --- CSS Styling ---
# This contains the styles for the entire application, including the navigation bar.
st.markdown("""
<style>
    /* --- Base Variables --- */
    :root {
        --font-main: 'sans-serif';
        --radius-m: 8px;
        --radius-l: 12px;
        --transition: all 0.3s ease;
    }

    /* --- Light Theme --- */
    body.light-mode {
        --primary-color: #4A90E2;
        --primary-color-light: #E8F1FC;
        --background-color: #F0F2F6;
        --secondary-background: #FFFFFF;
        --text-color: #333333;
        --text-color-subtle: #6A737D;
        --border-color: #DDE2E8;
    }

    /* --- Dark Theme --- */
    body.dark-mode {
        --primary-color: #58A6FF;
        --primary-color-light: #1A2D42;
        --background-color: #0E1117;
        --secondary-background: #161B22;
        --text-color: #EAEAEA;
        --text-color-subtle: #A0A0A0;
        --border-color: #30363D;
    }

    /* --- General Styles --- */
    body {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-main);
    }
    .stApp {
        background-color: var(--background-color);
    }
    h1, h2, h3 { color: var(--primary-color) !important; font-weight: 600; }
    p, li, label, .stMarkdown { color: var(--text-color); }

    /* --- Main Header --- */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0 0 0;
        color: var(--primary-color);
    }

    /* --- Page Sub-Header --- */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--primary-color);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid var(--border-color);
    }

    /* --- Card & Highlight Styles --- */
    .card {
        background-color: var(--secondary-background);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-l);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: var(--transition);
    }
    .highlight {
        background-color: var(--secondary-background);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-l);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
    }

    /* --- Custom Tabs --- */
    .nav-container {
        background-color: var(--secondary-background);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-l);
        padding: 10px;
        margin-bottom: 20px;
        display: flex;
        overflow-x: auto;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    
    .nav-container::-webkit-scrollbar {
        display: none;
    }
    
    /* Style for tab buttons */
    div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        border-radius: var(--radius-m);
        background-color: transparent;
        color: var(--text-color-subtle);
        border: none;
        font-weight: 500;
        padding: 8px 16px;
        transition: var(--transition);
    }
    
    div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        background-color: var(--primary-color-light);
        color: var(--primary-color);
    }
    
    div[data-testid="stHorizontalBlock"] button[kind="primary"] {
        border-radius: var(--radius-m);
        background-color: var(--primary-color);
        color: white;
        border: none;
        font-weight: 600;
        padding: 8px 16px;
        box-shadow: 0 4px 10px rgba(74, 144, 226, 0.3);
    }

    /* --- Mobile Responsiveness --- */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] button {
            padding: 6px 12px !important;
            font-size: 0.9rem !important;
        }
        
        .main-header {
            font-size: 1.8rem;
        }
        .sub-header {
            font-size: 1.4rem;
        }
        .card, .highlight {
            padding: 1rem;
        }
    }

    /* --- Footer --- */
    .footer {
        text-align: center;
        padding: 2rem;
        font-size: 0.9rem;
        color: var(--text-color-subtle);
    }
    .footer a {
        color: var(--primary-color) !important;
        text-decoration: none;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- HTML & JavaScript Components ---

# JS to apply the theme class to the body
components.html(f"""
<script>
    const body = window.parent.document.querySelector('body');
    body.classList.remove('light-mode', 'dark-mode');
    body.classList.add('{st.session_state.theme}-mode');
</script>""", height=0)

# --- Main Application ---
def main():
    """ Main function to run the Streamlit app. """
    # Header
    st.markdown('<h1 class="main-header">Computer Vision Explorer</h1>', unsafe_allow_html=True)
    
    # Theme toggle button
    col1, col2 = st.columns([0.95, 0.05])
    with col2:
        theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(theme_icon, key="theme_toggle"):
            toggle_theme()
    
    # Tab names and navigation
    tab_names = ["Home", "Feature Matching", "Homography", "Image Retrieval", "Epipolar Geometry", "Innovation"]
    
    # Create navigation container
    st.markdown('<div class="nav-container">', unsafe_allow_html=True)
    
    # Create tab buttons
    cols = st.columns(len(tab_names))
    for i, (col, name) in enumerate(zip(cols, tab_names)):
        with col:
            button_type = "primary" if i == st.session_state.current_tab else "secondary"
            st.button(name, key=f"tab_{i}", on_click=set_tab, args=(i,), use_container_width=True, type=button_type)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Render content based on the active tab index
    current_tab = st.session_state.current_tab
    
    if current_tab == 0:
        home.render_page()
    elif current_tab == 1:
        feature_matching.render_page()
    elif current_tab == 2:
        homography.render_page()
    elif current_tab == 3:
        image_retrieval.render_page()
    elif current_tab == 4:
        epipolar_geometry.render_page()
    elif current_tab == 5:
        innovation.render_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed by <a href="https://devshubh.me" target="_blank">Shubharthak Sangharasha</a> | 
        <a href="https://github.com/shubharthaksangharsha/trimester2/tree/main/opencv" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Ensure the 'pages' directory exists
    if not os.path.exists("pages"):
        os.makedirs("pages")
    # Ensure the __init__.py file exists
    if not os.path.exists("pages/__init__.py"):
        with open("pages/__init__.py", "w") as f:
            pass
            
    main()
