import streamlit as st
import pandas as pd
import gensim
from gensim import corpora
import pickle
import ast

# Load models (cached untuk performa)
@st.cache_resource
def load_models():
    models = {}
    dictionaries = {}
    
    # Load semua model
    models['negatif'] = gensim.models.LdaModel.load('lda_model_negatif')
    models['netral'] = gensim.models.LdaModel.load('lda_model_netral')
    models['positif'] = gensim.models.LdaModel.load('lda_model_positif')
    
    # Load dictionaries
    dictionaries['negatif'] = corpora.Dictionary.load('dictionary_negatif.dict')
    dictionaries['netral'] = corpora.Dictionary.load('dictionary_netral.dict')
    dictionaries['positif'] = corpora.Dictionary.load('dictionary_positif.dict')
    
    return models, dictionaries

# Load phraser models
@st.cache_resource  
def load_phrasers():
    import pickle
    with open('bigram_phraser.pkl', 'rb') as f:
        bigram_phraser = pickle.load(f)
    with open('trigram_phraser.pkl', 'rb') as f:
        trigram_phraser = pickle.load(f)
    return bigram_phraser, trigram_phraser

# UI
st.title("🔍 Topic Analysis Dashboard")
st.write("Analisis topik berdasarkan sentimen")

# Load models
models, dictionaries = load_models()
bigram_phraser, trigram_phraser = load_phrasers()

# Sidebar untuk pilih sentimen
sentiment = st.sidebar.selectbox(
    "Pilih Sentimen:",
    ['negatif', 'netral', 'positif']
)

# Tampilkan topik untuk sentimen yang dipilih
st.subheader(f"Topik untuk Sentimen {sentiment.title()}")

model = models[sentiment]
for idx, topic in model.print_topics():
    st.write(f"**Topik {idx+1}:**")
    st.write(topic)
    st.write("---")
