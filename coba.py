import streamlit as st
import pandas as pd
import gensim
from gensim import corpora
import pickle
import ast
import numpy as np

# Page config
st.set_page_config(
    page_title="Topic Analysis Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Load models (cached untuk performa)
@st.cache_resource
def load_models():
    try:
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
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load phraser models
@st.cache_resource  
def load_phrasers():
    try:
        with open('bigram_phraser.pkl', 'rb') as f:
            bigram_phraser = pickle.load(f)
        with open('trigram_phraser.pkl', 'rb') as f:
            trigram_phraser = pickle.load(f)
        return bigram_phraser, trigram_phraser
    except Exception as e:
        st.error(f"Error loading phrasers: {e}")
        return None, None

# Function untuk preprocess text baru
def preprocess_new_text(text_tokens, bigram_phraser, trigram_phraser):
    """Preprocess text dengan bigram dan trigram"""
    bigrams = bigram_phraser[text_tokens]
    trigrams = trigram_phraser[bigrams]
    return trigrams

# Function untuk prediksi topik
def predict_topics(text_tokens, sentiment, models, dictionaries, bigram_phraser, trigram_phraser):
    """Prediksi topik untuk text baru"""
    # Preprocess text
    processed_text = preprocess_new_text(text_tokens, bigram_phraser, trigram_phraser)
    
    # Convert ke bow
    bow = dictionaries[sentiment].doc2bow(processed_text)
    
    # Get topic distribution
    topic_dist = models[sentiment][bow]
    
    return topic_dist, processed_text

# Main app
def main():
    st.title("🔍 Topic Analysis Dashboard")
    st.markdown("---")
    
    # Load models
    models, dictionaries = load_models()
    bigram_phraser, trigram_phraser = load_phrasers()
    
    if models is None:
        st.error("⚠️ Models tidak berhasil dimuat!")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Konfigurasi")
    
    # Tab selection
    tab_mode = st.sidebar.radio(
        "Pilih Mode:",
        ["📊 Explore Topics", "📝 Analyze New Text", "📁 Upload Dataset"]
    )
    
    if tab_mode == "📊 Explore Topics":
        explore_topics_tab(models)
        
    elif tab_mode == "📝 Analyze New Text":
        analyze_text_tab(models, dictionaries, bigram_phraser, trigram_phraser)
        
    elif tab_mode == "📁 Upload Dataset":
        upload_dataset_tab(models, dictionaries, bigram_phraser, trigram_phraser)

def explore_topics_tab(models):
    """Tab untuk explore topics yang sudah ada"""
    st.header("📊 Explore Topics per Sentimen")
    
    # Pilih sentimen
    sentiment = st.selectbox(
        "Pilih Sentimen:",
        ['negatif', 'netral', 'positif'],
        key="explore_sentiment"
    )
    
    st.subheader(f"🔍 Topik untuk Sentimen {sentiment.title()}")
    
    model = models[sentiment]
    
    # Tampilkan info model
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Topik", model.num_topics)
    with col2:
        st.metric("Jumlah Kata dalam Vocab", len(model.id2word))
    
    # Tampilkan topik
    for idx, topic in model.print_topics(num_words=10):
        with st.expander(f"📝 Topik {idx+1}"):
            st.write(topic)

def analyze_text_tab(models, dictionaries, bigram_phraser, trigram_phraser):
    """Tab untuk analisis text baru"""
    st.header("📝 Analyze New Text")
    
    # Input text
    input_text = st.text_area(
        "Masukkan teks yang sudah di-tokenize (dalam format list):",
        value="['contoh', 'token', 'text']",
        height=100
    )
    
    # Pilih sentimen
    sentiment = st.selectbox(
        "Pilih Sentimen:",
        ['negatif', 'netral', 'positif'],
        key="analyze_sentiment"
    )
    
    if st.button("🔍 Analyze"):
        try:
            # Parse input text
            text_tokens = ast.literal_eval(input_text)
            
            # Prediksi topik
            topic_dist, processed_text = predict_topics(
                text_tokens, sentiment, models, dictionaries, 
                bigram_phraser, trigram_phraser
            )
            
            st.success("✅ Analisis berhasil!")
            
            # Tampilkan hasil
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Topic Distribution")
                if topic_dist:
                    for topic_id, prob in topic_dist:
                        st.write(f"**Topik {topic_id+1}:** {prob:.4f}")
                else:
                    st.write("Tidak ada topik yang cocok")
            
            with col2:
                st.subheader("🔤 Processed Text")
                st.write(processed_text)
                
        except Exception as e:
            st.error(f"Error: {e}")

def upload_dataset_tab(models, dictionaries, bigram_phraser, trigram_phraser):
    """Tab untuk upload dan analisis dataset"""
    st.header("📁 Upload & Analyze Dataset")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload file Excel/CSV:",
        type=['xlsx', 'csv']
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File berhasil diupload! Shape: {df.shape}")
            
            # Show preview
            st.subheader("👀 Preview Data")
            st.dataframe(df.head())
            
            # Column selection
            st.subheader("⚙️ Konfigurasi Kolom")
            
            col1, col2 = st.columns(2)
            with col1:
                text_column = st.selectbox(
                    "Pilih kolom text (tokenized):",
                    df.columns.tolist()
                )
            with col2:
                sentiment_column = st.selectbox(
                    "Pilih kolom sentimen:",
                    df.columns.tolist()
                )
            
            # Process dataset
            if st.button("🚀 Process Dataset"):
                process_uploaded_dataset(
                    df, text_column, sentiment_column,
                    models, dictionaries, bigram_phraser, trigram_phraser
                )
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

def process_uploaded_dataset(df, text_column, sentiment_column, models, dictionaries, bigram_phraser, trigram_phraser):
    """Process uploaded dataset"""
    try:
        # Convert text column to list if needed
        if isinstance(df[text_column].iloc[0], str):
            df[text_column] = df[text_column].apply(ast.literal_eval)
        
        # Progress bar
        progress_bar = st.progress(0)
        results = []
        
        for idx, row in df.iterrows():
            text_tokens = row[text_column]
            sentiment = row[sentiment_column].lower()
            
            # Map sentiment labels
            if sentiment in ['negatif', 'netral', 'positif']:
                try:
                    topic_dist, _ = predict_topics(
                        text_tokens, sentiment, models, dictionaries,
                        bigram_phraser, trigram_phraser
                    )
                    
                    # Get dominant topic
                    if topic_dist:
                        dominant_topic = max(topic_dist, key=lambda x: x[1])
                        results.append({
                            'index': idx,
                            'sentiment': sentiment,
                            'dominant_topic': dominant_topic[0] + 1,
                            'topic_prob': dominant_topic[1]
                        })
                    else:
                        results.append({
                            'index': idx,
                            'sentiment': sentiment,
                            'dominant_topic': 'No topic',
                            'topic_prob': 0
                        })
                except:
                    results.append({
                        'index': idx,
                        'sentiment': sentiment,
                        'dominant_topic': 'Error',
                        'topic_prob': 0
                    })
            
            # Update progress
            progress_bar.progress((idx + 1) / len(df))
        
        # Show results
        st.success("✅ Processing selesai!")
        
        results_df = pd.DataFrame(results)
        st.subheader("📊 Hasil Analisis Topic")
        st.dataframe(results_df)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="topic_analysis_results.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.subheader("📈 Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Data", len(results_df))
        with col2:
            avg_prob = results_df[results_df['topic_prob'] > 0]['topic_prob'].mean()
            st.metric("Avg Topic Probability", f"{avg_prob:.4f}" if not pd.isna(avg_prob) else "N/A")
        with col3:
            sentiment_counts = results_df['sentiment'].value_counts()
            st.write("**Sentiment Distribution:**")
            for sent, count in sentiment_counts.items():
                st.write(f"- {sent.title()}: {count}")
                
    except Exception as e:
        st.error(f"Error processing dataset: {e}")

if __name__ == "__main__":
    main()
