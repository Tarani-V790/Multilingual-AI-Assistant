import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from deep_translator import GoogleTranslator
from translate import Translator as OfflineTranslator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langdetect import detect, DetectorFactory
import base64
from PIL import Image
import json
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static

DetectorFactory.seed = 0
from gtts import gTTS
import os

def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
    os.system("start output.mp3")  # For Windows
import speech_recognition as sr

def listen_for_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return None
    except sr.RequestError:
        print("Could not request results, check your internet connection.")
        return None


# Import speech recognition and text-to-speech libraries
try:
    import speech_recognition as sr
    from gtts import gTTS
    from io import BytesIO
    import base64
    VOICE_FEATURES_AVAILABLE = True
except ImportError:
    VOICE_FEATURES_AVAILABLE = False

# Import Google API key from config
from config import GOOGLE_API_KEY

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Set the Streamlit page configuration and theme
st.set_page_config(
    page_title="RAYS Legal Assistant", 
    page_icon="⚖️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for language and history
if 'language_history' not in st.session_state:
    st.session_state.language_history = []

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"  # Default language

if 'language_selected' not in st.session_state:
    st.session_state.language_selected = False

# Automatic voice prompt for language selection
if VOICE_FEATURES_AVAILABLE:
    if not st.session_state.language_selected:
        # Speak the welcome message
        text_to_speech("Welcome to RAYS, TELL ME YOUR PREFERRED LANGUAGE", "en")
        
        # Listen for the user's preferred language
        with st.spinner("Listening for your preferred language..."):
            voice_input = listen_for_voice()
            if voice_input:
                # Update the selected language in the session state
                st.session_state.language_selected = True
                st.session_state.selected_language = voice_input
                st.success(f"Selected language: {voice_input}")
                
                # Refresh the UI to reflect the selected language
                st.rerun()


# Define custom CSS for a modern interface with enhanced design
def apply_custom_css():
    st.markdown("""
    <style>
        /* Modern Color Palette */
        :root {
            --primary-color: #4361ee;
            --primary-light: #4895ef;
            --secondary-color: #3a0ca3;
            --accent-color: #f72585;
            --success-color: #4cc9f0;
            --warning-color: #f8961e;
            --danger-color: #e63946;
            --background-color: #f8f9fa;
            --card-bg: #ffffff;
            --text-color: #2b2d42;
            --text-muted: #6c757d;
            --border-color: #e9ecef;
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1);
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
        }
        
        /* Global Styles */
        html, body, [class*="css"] {
            font-family: var(--font-sans);
            color: var(--text-color);
        }
        
        body {
            background-color: var(--background-color);
            background-image: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
            background-attachment: fixed;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 700;
            color: var(--secondary-color);
            letter-spacing: -0.025em;
        }
        
        h1 {
            font-size: 2.5rem;
            line-height: 1.2;
            margin-bottom: 1rem;
        }
        
        h2 {
            font-size: 2rem;
            line-height: 1.3;
            margin-bottom: 0.75rem;
        }
        
        h3 {
            font-size: 1.5rem;
            line-height: 1.4;
            margin-bottom: 0.5rem;
        }
        
        p {
            line-height: 1.6;
            margin-bottom: 1rem;
        }
        
        a {
            color: var(--primary-color);
            text-decoration: none;
            transition: color 0.2s ease;
        }
        
        a:hover {
            color: var(--primary-light);
            text-decoration: underline;
        }
        
        /* Main Header */
        .main-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem;
            border-radius: var(--radius-lg);
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMCAwIEwxMDAgMTAwIE0xMDAgMCBMMCAxMDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=');
            opacity: 0.1;
        }
        
        .main-header h1 {
            color: white;
            font-size: 2.75rem;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.75rem;
            max-width: 600px;
            margin-bottom: 0;
            margin-left:250px;
        }
        
        /* Cards and Containers */
        .feature-card {
            background-color: var(--card-bg);
            border-radius: var(--radius-md);
            padding: 1.5rem;
            box-shadow: var(--shadow-md);
            transition: all 0.3s ease;
            border: 1px solid var(--border-color);
            margin-bottom: 1.5rem;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
        }
        
        .feature-card h3 {
            color: var(--primary-color);
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .feature-card ul {
            padding-left: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .feature-card li {
            margin-bottom: 0.5rem;
        }
        
        /* Chat Messages */
        .user-message {
            background-color: #e7f5ff;
            border-left: 4px solid var(--primary-color);
            padding: 1rem 1.25rem;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            position: relative;
            animation: slideInRight 0.3s ease;
        }
        
        .assistant-message {
            background-color: #f8f9fa;
            border-left: 4px solid var(--success-color);
            padding: 1rem 1.25rem;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            margin-bottom: 1rem;
            box-shadow: var(--shadow-sm);
            position: relative;
            animation: slideInLeft 0.3s ease;
        }
        
        @keyframes slideInRight {
            from { transform: translateX(20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes slideInLeft {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        /* Buttons and Inputs */
        .stButton > button {
            background: linear-gradient(to right, var(--primary-color), var(--primary-light));
            color: white;
            border: none;
            border-radius: var(--radius-md);
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            letter-spacing: 0.025em;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-sm);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            background: linear-gradient(to right, var(--primary-light), var(--primary-color));
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Voice Button */
        .voice-button {
            background: linear-gradient(to right, var(--accent-color), #b5179e);
            color: white;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            box-shadow: var(--shadow-md);
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            margin: 0 auto;
        }
        
        .voice-button:hover {
            transform: scale(1.1);
            box-shadow: var(--shadow-lg);
        }
        
        .voice-button:active {
            transform: scale(0.95);
        }
        
        /* Input Fields */
        .stTextInput > div > div > input {
            border-radius: var(--radius-md);
            border: 2px solid var(--border-color);
            padding: 0.75rem 1rem;
            font-size: 1rem;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15);
        }
        
        /* Sidebar */
        .css-1d391kg, .css-1lcbmhc {
            background-color: #f8f9fa;
            border-right: 1px solid var(--border-color);
        }
        
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: var(--radius-md) var(--radius-md) 0 0;
            padding: 0.75rem 1.25rem;
            font-weight: 600;
            background-color: #f1f3f5;
            border: 1px solid var(--border-color);
            border-bottom: none;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white !important;
            color: var(--primary-color) !important;
            border-top: 3px solid var(--primary-color) !important;
        }
        
        /* Selectbox */
        .stSelectbox [data-baseweb="select"] > div:first-child {
            border-radius: var(--radius-md);
            border: 2px solid var(--border-color);
            transition: all 0.2s ease;
        }
        
        .stSelectbox [data-baseweb="select"] > div:first-child:focus-within {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(67, 97, 238, 0.15);
        }
        
        /* Toggle */
        .stCheckbox [data-baseweb="checkbox"] {
            gap: 0.5rem;
        }
        
        /* Progress Bars */
        .stProgress > div > div {
            background-color: var(--primary-light);
            border-radius: var(--radius-sm);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: var(--primary-color);
            background-color: #f8f9fa;
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            border: 1px solid var(--border-color);
        }
        
        .streamlit-expanderContent {
            border: 1px solid var(--border-color);
            border-top: none;
            border-radius: 0 0 var(--radius-md) var(--radius-md);
            padding: 1.25rem;
            background-color: white;
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
            color: white;
            padding: 2rem;
            border-radius: var(--radius-lg);
            margin-top: 3rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
            text-align: center;
        }
        
        .footer::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJNMCAwIEwxMDAgMTAwIE0xMDAgMCBMMCAxMDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjEpIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=');
            opacity: 0.1;
        }
        
        .footer p {
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 0.5rem;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .fade-in {
            animation: fadeIn 0.5s ease;
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
        }
        
        .typing-indicator span {
            width: 0.5rem;
            height: 0.5rem;
            background-color: var(--primary-color);
            border-radius: 50%;
            display: inline-block;
            animation: typing 1.4s infinite ease-in-out both;
        }
        
        .typing-indicator span:nth-child(1) {
            animation-delay: 0s;
        }
        
        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.6; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        /* Map Container */
        .map-container {
            border-radius: var(--radius-md);
            overflow: hidden;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
        }
        
        /* Legal Aid Cards */
        .legal-aid-card {
            background-color: white;
            border-radius: var(--radius-md);
            padding: 1.25rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
            margin-bottom: 1rem;
            display: flex;
            align-items: flex-start;
            gap: 1rem;
        }
        
        .legal-aid-card:hover {
            transform: translateY(-3px);
            box-shadow: var(--shadow-md);
        }
        
        .legal-aid-card .icon {
            background-color: #e7f5ff;
            color: var(--primary-color);
            width: 3rem;
            height: 3rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            flex-shrink: 0;
        }
        
        .legal-aid-card .content {
            flex-grow: 1;
        }
        
        .legal-aid-card h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            color: var(--primary-color);
        }
        
        /* Document Upload Area */
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: var(--radius-md);
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: var(--primary-color);
            background-color: #f0f4ff;
        }
        
        .upload-icon {
            font-size: 3rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .main-header {
                padding: 1.5rem;
            }
            
            .main-header h1 {
                font-size: 2rem;
            }
            
            .main-header p {
                font-size: 1rem;
            }
            
            .feature-card {
                padding: 1.25rem;
            }
            
            .voice-button {
                width: 50px;
                height: 50px;
                font-size: 1.25rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

if 'legal_centers' not in st.session_state:
    st.session_state.legal_centers = [
        {
            "name": "District Legal Services Authority",
            "contact": "+91 11 2345 6789",
            "lat": 28.6139,
            "lon": 77.2090
        },
        {
            "name": "National Legal Aid Clinic",
            "contact": "+91 11 8765 4321",
            "lat": 28.6329,
            "lon": 77.2195
        },
        {
            "name": "Women's Legal Support Center",
            "contact": "+91 11 9876 5432",
            "lat": 28.6508,
            "lon": 77.2359
        }
    ]

def get_user_location(place_name):
    try:
        geolocator = Nominatim(user_agent="Rays_legal_assistant")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
        return None
    except:
        return None

# Function to create map with nearby legal aid centers
def create_legal_aid_map(user_lat, user_lon):
    m = folium.Map(location=[user_lat, user_lon], zoom_start=12, tiles="CartoDB positron")
    
    # Add user marker
    folium.Marker(
        location=[user_lat, user_lon],
        popup="Your Location",
        icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    ).add_to(m)
    
    # Add markers for legal aid centers
    for center in st.session_state.legal_centers:
        folium.Marker(
            location=[center["lat"], center["lon"]],
            popup=f"<b>{center['name']}</b><br>Contact: {center['contact']}",
            icon=folium.Icon(color="green", icon="balance-scale", prefix="fa"),
        ).add_to(m)
    
    # Add circle around user location
    folium.Circle(
        location=[user_lat, user_lon],
        radius=3000,  # 3km radius
        color="#4361ee",
        fill=True,
        fill_color="#4361ee",
        fill_opacity=0.1
    ).add_to(m)
    
    return m

# Function to translate text dynamically
def translate_text(text, target_language):
    try:
        # Attempt online translation
        translated_text = GoogleTranslator(source='auto', target=target_language.lower()).translate(text)
        return translated_text
    except Exception as e:
        st.warning("Online translation failed, attempting offline translation.")
        try:
            # Attempt offline translation
            offline_translator = OfflineTranslator(to_lang=target_language.lower())
            translated_text = offline_translator.translate(text)
            return translated_text
        except Exception as e:
            st.error(f"Translation failed: {str(e)}")
            return text  # Return original text if translation fails

# Function to play text-to-speech
def text_to_speech(text, lang_code="en"):
    language_codes = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Punjabi": "pa",
        "Urdu": "ur",
        "Assamese": "as",
        "Odia": "or",
        "Nepali": "ne",
        "Sindhi": "sd"
    }
    
    try:
        tts = gTTS(text=text, lang=language_codes.get(lang_code, "en"), slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f'<audio autoplay="true" src="data:audio/mp3;base64,{audio_base64}"></audio>'
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")

# Function to listen to voice input
def listen_for_voice():
    if not VOICE_FEATURES_AVAILABLE:
        return None

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.markdown('<div class="assistant-message"><div class="typing-indicator"><span></span><span></span><span></span></div> Listening... Please speak your query.</div>', unsafe_allow_html=True)
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)
        
    try:
        language_codes = {
            "English": "en-US",
            "Hindi": "hi-IN",
            "Tamil": "ta-IN",
            "Telugu": "te-IN",
            "Kannada": "kn-IN", 
            "Malayalam": "ml-IN",
            "Bengali": "bn-IN",
            "Gujarati": "gu-IN",
            "Marathi": "mr-IN", 
            "Punjabi": "pa-IN",
            "Urdu": "ur-IN",
            "Assamese": "as-IN",
            "Odia": "or-IN",
            "Nepali": "ne-NP",
            "Sindhi": "sd"
        }
        
        lang_code = language_codes.get(st.session_state.selected_language, "en-US")
        text = recognizer.recognize_google(audio, language=lang_code)
        return text
    except sr.UnknownValueError:
        st.warning("Sorry, I couldn't understand what you said.")
    except sr.RequestError:
        st.error("Could not request results; check your network connection")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    return None
if VOICE_FEATURES_AVAILABLE:
    if not st.session_state.language_selected:
        # Speak the welcome message
        text_to_speech("Welcome to RAYS, TELL ME YOUR PREFERRED LANGUAGE", "en")
        
        # Listen for the user's preferred language
        with st.spinner("Listening for your preferred language..."):
            voice_input = listen_for_voice()
            if voice_input:
                # Update the selected language in the session state
                st.session_state.language_selected = True
                st.session_state.selected_language = voice_input
                st.success(f"Selected language: {voice_input}")
                
                # Refresh the UI to reflect the selected language
                st.experimental_rerun()
# Sidebar configuration with improved styling
with st.sidebar:
    st.markdown('<h1 style="color: #4361ee; display: flex; align-items: center; gap: 0.5rem;"><span>⚖️</span> R.A.Y.S</h1>', unsafe_allow_html=True)
    
    # Language selection with improved styling
    
    selected_language = st.selectbox(
        "Select your preferred language",
        ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"],
        index=0 if 'selected_language' not in st.session_state else ["English", "Assamese", "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi", "Nepali", "Odia", "Punjabi", "Sindhi", "Tamil", "Telugu", "Urdu"].index(st.session_state.selected_language)
    )

    # Update session state with the selected language
    st.session_state.selected_language = selected_language

    if selected_language not in st.session_state.language_history:
        st.session_state.language_history.append(selected_language)


    # Translate sidebar content
    st.markdown(f"""
    <div class="feature-card">
        <h3>{translate_text("🌟 About RAYS",selected_language)}</h3>
        <p>{translate_text("RAYS is designed to make legal assistance accessible to all, especially in rural and underserved communities. Get accurate legal information in your preferred language through text or voice interaction.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Mode toggle with improved styling
    st.markdown(f'<div style="margin-bottom: 1rem;"><h3 style="margin-bottom: 0.5rem;">⚙️ {translate_text("Settings", selected_language)}</h3></div>', unsafe_allow_html=True)

    model_mode = st.toggle(translate_text("Online Mode", selected_language), value=True, help="Toggle between online and offline mode")

    # Voice features toggle
    if VOICE_FEATURES_AVAILABLE:
        voice_enabled = st.toggle(translate_text("Enable Voice Interaction", selected_language), value=True, help="Enable voice input and text-to-speech output")
        if voice_enabled:
            st.markdown(f'<p style="color: #4361ee;"><i>🎙 {translate_text("Voice interaction is enabled. Click the microphone button to speak your query.", selected_language)}</i></p>', unsafe_allow_html=True)
    else:
        voice_enabled = False
        st.warning(translate_text("Voice libraries not installed. Run: pip install SpeechRecognition gtts", selected_language))

    # Legal resources section with improved styling
    st.markdown(f"""
    <div class="feature-card">
        <h3>📚 {translate_text("Legal Resources", selected_language)}</h3>
        <ul>
            <li><a href="https://www.legalservicesindia.com/" target="_blank">{translate_text("Legal Services India", selected_language)}</a></li>
            <li><a href="https://nalsa.gov.in/" target="_blank">{translate_text("National Legal Services Authority", selected_language)}</a></li>
            <li><a href="https://doj.gov.in/" target="_blank">{translate_text("Department of Justice", selected_language)}</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


    # Emergency contacts with improved styling
    st.markdown(f"""
    <div class="feature-card">
        <h3>{translate_text("🆘 Emergency Contacts",selected_language)}</h3>
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="background-color: #f1f3f5; border-radius: 50%; width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; color: #4361ee;">📞</div>
            <div>
                <strong>{translate_text("National Legal Aid Helpline:", selected_language)}</strong> 15100
            </div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <div style="background-color: #f1f3f5; border-radius: 50%; width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; color: #4361ee;">📞</div>
            <div>
                <strong>{translate_text("Women Helpline:", selected_language)}</strong> 1091
            </div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: #f1f3f5; border-radius: 50%; width: 2rem; height: 2rem; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem; color: #4361ee;">📞</div>
            <div>
                <strong>{translate_text("Child Helpline:", selected_language)}</strong> 1098
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Main content with improved header
st.markdown(f'''
<div class="main-header">
        <p>{translate_text("Rights Assistance for Youth and Society", selected_language)}</p>

</div>
''', unsafe_allow_html=True)

# Tabs for different functionalities with improved styling
tabs = st.tabs([
    f"🗺 {translate_text('Legal Aid Locator', selected_language)}",
    f"💬 {translate_text('Chat Assistant', selected_language)}",
    f"📊 {translate_text('Legal Awareness', selected_language)}",
    f"❓ {translate_text('FAQ', selected_language)}",
    # f"📄 {translate_text('Document Verification', selected_language)}",
    # f"📚 {translate_text('Legal Database', selected_language)}",
])

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Load and process your text data (Replace this with your actual legal text data)
text_data = """
[Your legal text data here]
"""

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(text_data)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

# Convert vector store into a retriever
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Continue with the rest of the app...
db_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
with tabs[0]:  # Legal Aid Locator Tab
    st.markdown(f"""
    <div class="feature-card fade-in">
        <h2>{translate_text("🗺 Find Legal Aid Centers Near You", selected_language)}</h2>
        <p>{translate_text("Enter your location to find legal aid centers and free legal services available in your area.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for input and map
    loc_col1, loc_col2 = st.columns([1, 1])
    
    with loc_col1:
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        location_input = st.text_input(translate_text("Enter your city or area", selected_language), "Delhi")
        if st.button(translate_text("Find Legal Aid Centers", selected_language), key="find_centers"):
            with st.spinner(translate_text("Locating centers...", selected_language)):
                user_location = get_user_location(location_input)
                if user_location:
                    user_lat, user_lon = user_location
                    st.session_state.user_location = user_location
                    st.success(f"📍 {translate_text('Location found', selected_language)}: {location_input}")
                else:
                    st.error(translate_text("Location not found. Please try a different city or check your spelling.", selected_language))
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display eligibility information
        st.markdown(f"""
        <div class="feature-card fade-in">
            <h3>✅ {translate_text("Free Legal Aid Eligibility", selected_language)}</h3>
            <p>{translate_text("In India, free legal services are available to:", selected_language)}</p>
            <ul>
                <li>{translate_text("Women and children", selected_language)}</li>
                <li>{translate_text("Victims of trafficking", selected_language)}</li>
                <li>{translate_text("Persons with disabilities", selected_language)}</li>
                <li>{translate_text("Victims of mass disaster, ethnic violence, caste atrocity, flood, drought, earthquake, industrial disaster", selected_language)}</li>
                <li>{translate_text("Industrial workmen", selected_language)}</li>
                <li>{translate_text("Persons in custody", selected_language)}</li>
                <li>{translate_text("Persons with annual income less than Rs. 1,00,000 (may vary by state)", selected_language)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with loc_col2:
        st.markdown('<div class="map-container fade-in">', unsafe_allow_html=True)
        if 'user_location' in st.session_state:
            user_lat, user_lon = st.session_state.user_location
            legal_map = create_legal_aid_map(user_lat, user_lon)
            st.markdown(f"<h3>📍 {translate_text('Legal Aid Centers Near', selected_language)} {location_input}</h3>", unsafe_allow_html=True)
            folium_static(legal_map)
        else:
            st.markdown(f"<h3>📍 {translate_text('Map will appear here', selected_language)}</h3>", unsafe_allow_html=True)
            # Display a placeholder map centered on India
            m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="CartoDB positron")
            folium_static(m)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display legal aid centers in a modern card layout
    if 'user_location' in st.session_state:
        st.markdown(f"<h3>📋 {translate_text('Available Legal Aid Centers', selected_language)}</h3>", unsafe_allow_html=True)
        
        for center in st.session_state.legal_centers:
            st.markdown(f"""
            <div class="legal-aid-card fade-in">
                <div class="icon">⚖️</div>
                <div class="content">
                    <h4>{center['name']}</h4>
                    <p><strong>{translate_text("Contact:", selected_language)}</strong> {center['contact']}</p>
                    <p><strong>{translate_text("Services:", selected_language)}</strong> {translate_text("Free legal advice, document preparation, court representation", selected_language)}</p>
                    <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
                        <a href="tel:{center['contact'].replace(' ', '')}" style="text-decoration: none;">
                            <button style="background-color: #4361ee; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: 500; cursor: pointer;">
                                {translate_text("Call Now", selected_language)}
                            </button>
                        </a>
                        <a href="https://maps.google.com/?q={center['lat']},{center['lon']}" target="_blank" style="text-decoration: none;">
                            <button style="background-color: #f8f9fa; color: #4361ee; border: 1px solid #4361ee; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: 500; cursor: pointer;">
                                {translate_text("Get Directions", selected_language)}
                            </button>
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tabs[1]:  # Chat Assistant Tab
    # Create two columns for chat and voice input
    chat_col, voice_col = st.columns([5, 1])

    # Display chat messages with improved styling
    st.markdown('<div class="chat-container fade-in">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message"><strong>RAYS:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    def get_response_online(prompt, context):
        full_prompt = f"""
        As a legal chatbot specializing in the Indian Penal Code and Department of Justice services, you are tasked with providing highly accurate and contextually appropriate responses. Ensure your answers meet these criteria:
        - Respond in a bullet-point format to clearly delineate distinct aspects of the legal query or service information.
        - Each point should accurately reflect the breadth of the legal provision or service in question, avoiding over-specificity unless directly relevant to the user's query.
        - Clarify the general applicability of the legal rules, sections, or services mentioned, highlighting any common misconceptions or frequently misunderstood aspects.
        - Limit responses to essential information that directly addresses the user's question, providing concise yet comprehensive explanations.
        - When asked about live streaming of court cases, provide the relevant links for court live streams.
        - For queries about various DoJ services or information, provide accurate links and guidance.
        - Avoid assuming specific contexts or details not provided in the query, focusing on delivering universally applicable legal interpretations or service information unless otherwise specified.
        - Conclude with a brief summary that captures the essence of the legal discussion or service information and corrects any common misinterpretations related to the topic.
        - When providing legal information, also mention if free legal aid may be available for the situation.
        - If asked about legal aid centers, mention that users can check the Legal Aid Locator tab.

        CONTEXT: {context}
        QUESTION: {prompt}
        ANSWER:
        """
        response = model.generate_content(full_prompt, stream=True)
        return response

    def get_response_offline(prompt, context):
        llm = ChatOllama(model="phi3")
        # Implement offline response generation here
        # This is a placeholder and needs to be implemented based on your offline requirements
        return "Offline mode is not fully implemented yet."

    def translate_answer(answer, target_language):
        try:
            # Attempt online translation
            translated_answer = GoogleTranslator(source='auto', target=target_language.lower()).translate(answer)
            return translated_answer
        except Exception as e:
            st.warning("Online translation failed, attempting offline translation.")
            try:
                # Attempt offline translation
                offline_translator = OfflineTranslator(to_lang=target_language.lower())
                translated_answer = offline_translator.translate(answer)
                return translated_answer
            except Exception as e:
                st.error(f"Offline translation failed: {str(e)}")
                return answer 

    def reset_conversation():
        st.session_state.messages = []
        st.session_state.memory.clear()

    def get_trimmed_chat_history():
        max_history = 10
        return st.session_state.messages[-max_history:]

    # Voice input button in the voice column
    with voice_col:
        if VOICE_FEATURES_AVAILABLE and voice_enabled:
            st.markdown('<div style="display: flex; justify-content: center; margin-top: 1rem;">', unsafe_allow_html=True)
            if st.button("🎤", help="Click to speak your query", key="voice_button"):
                with st.spinner("Listening..."):
                    voice_input = listen_for_voice()
                    if voice_input:
                        st.session_state.voice_input = voice_input
            st.markdown('</div>', unsafe_allow_html=True)

    # Handle user input (either from text or voice)
    input_prompt = None
    
    # Check if there's voice input in session state
    if hasattr(st.session_state, 'voice_input') and st.session_state.voice_input:
        input_prompt = st.session_state.voice_input
        st.session_state.voice_input = None  # Clear after use
    else:
        # Regular text input with improved styling
        input_prompt = st.chat_input(translate_text("Start with your legal query", selected_language), key="chat_input")
    
    if input_prompt:
        st.markdown(f'<div class="user-message fade-in"><strong>You:</strong> {input_prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "user", "content": input_prompt})
        trimmed_history = get_trimmed_chat_history()

        with st.spinner("Thinking 💡..."):
            context = db_retriever.get_relevant_documents(input_prompt)
            context_text = "\n".join([doc.page_content for doc in context])
            
            if model_mode:
                response = get_response_online(input_prompt, context_text)
            else:
                response = get_response_offline(input_prompt, context_text)

            message_placeholder = st.empty()
            full_response = "Gentle reminder: We generally ensure precise information, but do double-check. \n\n\n"
            
            if model_mode:
                for chunk in response:
                    full_response += chunk.text
                    time.sleep(0.02)  # Adjust the sleep time to control the "typing" speed
                    message_placeholder.markdown(f'<div class="assistant-message fade-in"><strong>RAYS:</strong> {full_response}</div>', unsafe_allow_html=True)
            else:
                full_response += response
                message_placeholder.markdown(f'<div class="assistant-message fade-in"><strong>RAYS:</strong> {full_response}</div>', unsafe_allow_html=True)

            # Translate the answer to the selected language
            if selected_language != "English":
                with st.spinner(f"Translating to {selected_language}..."):
                    translated_answer = translate_answer(full_response, selected_language.lower())
                    message_placeholder.markdown(f'<div class="assistant-message fade-in"><strong>RAYS:</strong> {translated_answer}</div>', unsafe_allow_html=True)
                    
                    # Play TTS for the translated response if voice is enabled
                    if VOICE_FEATURES_AVAILABLE and voice_enabled:
                        text_to_speech(translated_answer, selected_language.lower())
            else:
                # Play TTS for English response if voice is enabled
                if VOICE_FEATURES_AVAILABLE and voice_enabled:
                    text_to_speech(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Reset button with improved styling
    st.markdown('<div style="display: flex; justify-content: center; margin-top: 1.5rem;">', unsafe_allow_html=True)
    if st.button(translate_text('🗑 Reset Conversation', selected_language), on_click=reset_conversation, key="reset_button"):
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:  # Legal Awareness Tab
    st.markdown(f"""
    <div class="feature-card fade-in">
        <h2>{translate_text("📊 Legal Awareness", selected_language)}</h2>
        <p>{translate_text("Educational resources to understand your legal rights and responsibilities.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Translate legal topics
    legal_topics = [
        translate_text("Women's Rights", selected_language),
        translate_text("Property Laws", selected_language),
        translate_text("Consumer Protection", selected_language),
        translate_text("Labor Laws", selected_language),
        translate_text("Right to Information", selected_language)
    ]
    
    # Create two columns for topic selection and content
    topic_col1, topic_col2 = st.columns([1, 3])
    
    with topic_col1:
        # Translate the selectbox label
        selected_topic = st.selectbox(translate_text("Select a topic", selected_language), legal_topics)
    
    # Translate topic content dynamically
    topic_content = {
        translate_text("Women's Rights", selected_language): f"""
        <div class="feature-card fade-in">
            <h3>📝 {translate_text("Women's Legal Rights in India", selected_language)}</h3>
            <ul>
                <li><strong>{translate_text("Protection from Domestic Violence:", selected_language)}</strong> {translate_text("Under the Protection of Women from Domestic Violence Act, 2005.", selected_language)}</li>
                <li><strong>{translate_text("Equal Pay:", selected_language)}</strong> {translate_text("The Equal Remuneration Act, 1976 mandates equal pay for equal work.", selected_language)}</li>
                <li><strong>{translate_text("Maternity Benefits:", selected_language)}</strong> {translate_text("The Maternity Benefit Act provides for 26 weeks of paid maternity leave.", selected_language)}</li>
                <li><strong>{translate_text("Protection at Workplace:", selected_language)}</strong> {translate_text("Sexual Harassment of Women at Workplace (Prevention, Prohibition and Redressal) Act, 2013.", selected_language)}</li>
                <li><strong>{translate_text("Property Rights:", selected_language)}</strong> {translate_text("Equal inheritance rights under the Hindu Succession (Amendment) Act, 2005.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Helplines:", selected_language)}</strong> {translate_text("Women's Helpline: 1091, Domestic Abuse Helpline: 181", selected_language)}</p>
        </div>
        """,
        
        translate_text("Property Laws", selected_language): f"""
        <div class="feature-card fade-in">
            <h3>📝 {translate_text("Property Laws - Key Points", selected_language)}</h3>
            <ul>
                <li><strong>{translate_text("Registration:", selected_language)}</strong> {translate_text("All property transactions should be registered under the Registration Act, 1908.", selected_language)}</li>
                <li><strong>{translate_text("Stamp Duty:", selected_language)}</strong> {translate_text("Mandatory payment varying by state (typically 5-10% of property value).", selected_language)}</li>
                <li><strong>{translate_text("Inheritance:", selected_language)}</strong> {translate_text("Governed by personal laws (Hindu, Muslim, Christian, Parsi) or Indian Succession Act.", selected_language)}</li>
                <li><strong>{translate_text("Tenant Rights:", selected_language)}</strong> {translate_text("Protected under various Rent Control Acts in different states.", selected_language)}</li>
                <li><strong>{translate_text("Land Ceiling:", selected_language)}</strong> {translate_text("Restrictions on maximum land holdings in urban areas.", selected_language)}</li>
            </ul>
        </div>
        """,
        
        translate_text("Consumer Protection", selected_language): f"""
        <div class="feature-card fade-in">
            <h3>📝 {translate_text("Consumer Protection Rights", selected_language)}</h3>
            <p>{translate_text("Under the Consumer Protection Act, 2019, you have the right to:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("Right to Safety:", selected_language)}</strong> {translate_text("Protection against hazardous goods and services.", selected_language)}</li>
                <li><strong>{translate_text("Right to Information:", selected_language)}</strong> {translate_text("Complete details about performance, quality, quantity, and price.", selected_language)}</li>
                <li><strong>{translate_text("Right to Choose:", selected_language)}</strong> {translate_text("Access to variety of goods at competitive prices.", selected_language)}</li>
                <li><strong>{translate_text("Right to be Heard:", selected_language)}</strong> {translate_text("Have your interests receive due consideration.", selected_language)}</li>
                <li><strong>{translate_text("Right to Redressal:", selected_language)}</strong> {translate_text("Fair settlement of genuine grievances.", selected_language)}</li>
                <li><strong>{translate_text("Right to Consumer Education:", selected_language)}</strong> {translate_text("Acquire knowledge and skills to be an informed consumer.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("How to File a Complaint:", selected_language)}</strong> {translate_text("Visit the nearest Consumer Forum or file online at", selected_language)} <a href="https://consumerhelpline.gov.in" target="_blank">{translate_text("National Consumer Helpline", selected_language)}</a></p>
        </div>
        """,
        
        translate_text("Labor Laws", selected_language): f"""
        <div class="feature-card fade-in">
            <h3>📝 {translate_text("Key Labor Laws in India", selected_language)}</h3>
            <ul>
                <li><strong>{translate_text("Minimum Wages Act, 1948:", selected_language)}</strong> {translate_text("Ensures minimum wage payment to workers.", selected_language)}</li>
                <li><strong>{translate_text("Factories Act, 1948:", selected_language)}</strong> {translate_text("Regulates working conditions in factories.", selected_language)}</li>
                <li><strong>{translate_text("Payment of Gratuity Act, 1972:", selected_language)}</strong> {translate_text("Provides for gratuity payment to employees.", selected_language)}</li>
                <li><strong>{translate_text("Employees' Provident Fund Act:", selected_language)}</strong> {translate_text("Ensures retirement benefits.", selected_language)}</li>
                <li><strong>{translate_text("Payment of Bonus Act, 1965:", selected_language)}</strong> {translate_text("Provides for annual bonus payment.", selected_language)}</li>
                <li><strong>{translate_text("Industrial Disputes Act, 1947:", selected_language)}</strong> {translate_text("Mechanism for settlement of industrial disputes.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("For Labor Disputes:", selected_language)}</strong> {translate_text("Contact your local Labor Commissioner Office", selected_language)}</p>
        </div>
        """,
        
        translate_text("Right to Information", selected_language): f"""
        <div class="feature-card fade-in">
            <h3>📝 {translate_text("Right to Information (RTI) Act, 2005", selected_language)}</h3>
            <p><strong>{translate_text("What is RTI?", selected_language)}</strong> {translate_text("A law that allows citizens to request information from any public authority.", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("How to File an RTI:", selected_language)}</strong> {translate_text("Submit application with Rs. 10 fee to the Public Information Officer (PIO).", selected_language)}</li>
                <li><strong>{translate_text("Time Limit:", selected_language)}</strong> {translate_text("Information must be provided within 30 days (48 hours if life/liberty is involved).", selected_language)}</li>
                <li><strong>{translate_text("Appeal Process:", selected_language)}</strong> {translate_text("First appeal to designated officer, second appeal to Information Commission.", selected_language)}</li>
                <li><strong>{translate_text("Exemptions:", selected_language)}</strong> {translate_text("Information affecting sovereignty, security, strategic interests, trade secrets, privacy, etc.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Online RTI Filing:", selected_language)}</strong> <a href="https://rtionline.gov.in" target="_blank">{translate_text("RTI Online Portal", selected_language)}</a></p>
        </div>
        """
    }
    
    st.markdown(f"""
    <div class="feature-card fade-in">
        <h3>📺 {translate_text("Educational Videos", selected_language)}</h3>
        <p>{translate_text("Watch informative videos to better understand legal concepts.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)

    # Column Layout for Videos
    video_col1, video_col2 = st.columns(2)

    with video_col1:
        st.video("https://youtu.be/KAGWjGzo-28")  #
    with video_col2:
        st.video("https://youtu.be/OIsyVZCB3KM") 

with tabs[3]:  # FAQ Tab
    st.markdown(f"""
    <div class="feature-card fade-in">
        <h2>{translate_text("❓ Frequently Asked Questions",selected_language)}</h2>
        <p>{translate_text("Find answers to common legal questions and concerns.", selected_language)}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add state selection dropdown with improved styling
    state_legal_aid = {
        "Maharashtra": {
            "legal_aid_authority": "Maharashtra State Legal Services Authority (MSLSA)",
            "helpline": "1800-22-1117",
            "website": "https://legalservices.maharashtra.gov.in",
            "specific_laws": "Maharashtra Rent Control Act, Maharashtra Police Act"
        },
        "Delhi": {
            "legal_aid_authority": "Delhi State Legal Services Authority (DSLSA)",
            "helpline": "1516",
            "website": "https://dslsa.org",
            "specific_laws": "Delhi Rent Control Act, Delhi Maintenance and Welfare of Parents and Senior Citizens Rules"
        },
        "Tamil Nadu": {
            "legal_aid_authority": "Tamil Nadu State Legal Services Authority (TNSLSA)",
            "helpline": "044-25342441",
            "website": "https://tnslsa.tn.gov.in",
            "specific_laws": "Tamil Nadu Apartment Ownership Act, Tamil Nadu Hindu Religious and Charitable Endowments Act"
        }
    }

    # Default information for states not explicitly defined
    default_legal_aid = {
        "legal_aid_authority": "State Legal Services Authority",
        "helpline": "15100 (NALSA Helpline)",
        "website": "https://nalsa.gov.in",
        "specific_laws": "Local State Acts and Regulations"
    }

    # Ensure selected_state is initialized before use
    selected_state = st.selectbox(
        "Select State/UT for specific legal information:",
        ["All India", "Maharashtra", "Delhi", "Tamil Nadu"]
    )
  
    # Get state-specific information or default if not available
    state_info = state_legal_aid.get(selected_state, default_legal_aid)
    
    # Display state-specific note if a specific state is selected
    if selected_state != "All India":
        st.info(translate_text(f"Showing specific legal information for {selected_state}. Some details may vary by district.", selected_language))
    
    # FAQ Accordion with state-specific information and improved styling
    with st.expander(translate_text("1. How can I get free legal aid in India?", selected_language)):
        st.markdown(f"""
        <div class="feature-card fade-in">
            <p>{translate_text("Free legal aid is available through the", selected_language)} <strong>{translate_text(state_info["legal_aid_authority"], selected_language)}</strong> {translate_text("and its district-level counterparts. You can:", selected_language)}</p>
            <ul>
                <li>{translate_text("Visit your nearest", selected_language)} <strong>{translate_text("District Legal Services Authority (DLSA)", selected_language)}</strong>.</li>
                <li>{translate_text("Call the helpline at", selected_language)} <strong>{state_info["helpline"]}</strong>.</li>
                <li>{translate_text("Apply online through the", selected_language)} <a href="{state_info["website"]}" target="_blank">{translate_text("official website", selected_language)}</a>.</li>
            </ul>
          
            <p><strong>{translate_text("Eligibility:", selected_language)}</strong> {translate_text("Women, children, SC/ST communities, victims of trafficking, and individuals with an annual income below ₹1,00,000 are eligible.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("2. What should I do if I'm a victim of domestic violence?", selected_language)):
        # State-specific domestic violence helplines could be added here
        domestic_violence_helplines = {
            "Maharashtra": "103",
            "Delhi": "181",
            "Tamil Nadu": "181"
        }.get(selected_state, "181")
        
        st.markdown(f"""
        <div class="feature-card fade-in">
            <p>{translate_text("If you're facing domestic violence, take the following steps:", selected_language)}</p>
            <ul>
                <li>{translate_text("Call the", selected_language)} <strong>{translate_text("Women's Helpline", selected_language)}</strong> {translate_text("at", selected_language)} <strong>{domestic_violence_helplines}</strong> {translate_text("or", selected_language)} <strong>1091</strong>.</li>
                <li>{translate_text("File a complaint under the", selected_language)} <strong>{translate_text("Protection of Women from Domestic Violence Act, 2005", selected_language)}</strong>.</li>
                <li>{translate_text("Contact a", selected_language)} <strong>{translate_text("Protection Officer", selected_language)}</strong> {translate_text("in your district.", selected_language)}</li>
                <li>{translate_text("Seek help from local NGOs and women's support groups.", selected_language)}</li>
            </ul>
            <p><strong>{translate_text("Note:", selected_language)}</strong> {translate_text("You can also approach the nearest police station or family court for immediate assistance.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(translate_text("3. What are my rights as a tenant?", selected_language)):
        st.markdown(f"""
        <div class="feature-card fade-in">
            <p>{translate_text("As a tenant in", selected_language)} {selected_state}, {translate_text("you have the following rights:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("Right to a Rent Agreement:", selected_language)}</strong> {translate_text("Ensure you have a written agreement.", selected_language)}</li>
                <li><strong>{translate_text("Protection from Eviction:", selected_language)}</strong> {translate_text("Landlords cannot evict you without proper notice.", selected_language)}</li>
                <li><strong>{translate_text("Right to Essential Services:", selected_language)}</strong> {translate_text("Landlords must provide water, electricity, and maintenance.", selected_language)}</li>
                <li><strong>{translate_text("Security Deposit:", selected_language)}</strong> {translate_text("You are entitled to the return of your deposit upon vacating.", selected_language)}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Continue with other FAQ items, adding state-specific information where relevant
    with st.expander(translate_text("4. How do I file an RTI application?", selected_language)):
        # State-specific RTI portals
        rti_portals = {
            "Maharashtra": "https://maharashtra.rti.gov.in",
            "Delhi": "https://rtionline.delhi.gov.in",
            "Tamil Nadu": "https://rti.tn.gov.in"
        }.get(selected_state, "https://rtionline.gov.in")
        
        st.markdown(f"""
        <div class="feature-card fade-in">
            <p>{translate_text("To file an RTI application in", selected_language)} {selected_state}:</p>
            <ul>
                <li>{translate_text("Write a clear application stating the information you need.", selected_language)}</li>
                <li>{translate_text("Pay a fee of ₹10 (waived for below-poverty-line applicants).", selected_language)}</li>
                <li>{translate_text("Submit the application to the", selected_language)} <strong>{translate_text("Public Information Officer (PIO)", selected_language)}</strong> {translate_text("of the relevant department.", selected_language)}</li>
                <li>{translate_text("You can file online at the", selected_language)} <a href="{rti_portals}" target="_blank">{translate_text("RTI Portal", selected_language)}</a>.</li>
            </ul>
            <p><strong>{translate_text("Response Time:", selected_language)}</strong> {translate_text("Information must be provided within 30 days.", selected_language)}</p>
        </div>
        """, unsafe_allow_html=True)

# Add more state-specific laws and regulations section
if selected_state != "All India":
    with st.expander(translate_text(f"5. What are the specific laws and regulations in {selected_state}?", selected_language)):
        st.markdown(f"""
        <div class="feature-card fade-in">
            <p>{translate_text(f"Important laws and regulations specific to {selected_state}:", selected_language)}</p>
            <ul>
                <li><strong>{translate_text("State-Specific Laws:", selected_language)}</strong> {translate_text(state_info["specific_laws"], selected_language)}</li>
                <li><strong>{translate_text("Legal Services Authority:", selected_language)}</strong> {translate_text(state_info["legal_aid_authority"], selected_language)}</li>
            </ul>
            <p>{translate_text("For detailed information about specific laws in your state, please visit the", selected_language)} 
            <a href="{state_info["website"]}" target="_blank">{translate_text("official website", selected_language)}</a>.</p>
        </div>
        """, unsafe_allow_html=True)

        # Closing div (if needed)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<footer style="text-align: center; padding: 1rem; background-color: #f8f9fa; border-top: 1px solid var(--border-color);">
    <p style="color: var(--text-muted);">© 2023 RAYS Legal Assistant. All rights reserved.</p>
</footer>
""", unsafe_allow_html=True)