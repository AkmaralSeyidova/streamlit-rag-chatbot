import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatHuggingFace
from langchain.llms import HuggingFaceEndpoint
from langchain.hub import pull
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
import os
import tempfile

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")

# API key check
if not hf_api_key:
    st.error("❌ Hugging Face API key is missing. Please configure it in the .env file or Streamlit Secrets.")
    st.stop()

# Streamlit UI setup
st.set_page_config(page_title="RAG Document Q&A", layout="centered")

st.markdown("""
<style>
.stApp {
    background-color: #f0f8ff;
    font-family: Arial, sans-serif;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    margin-top: 20px;
}
.response {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 10px;
    margin-top: 20px;
    font-size: 16px;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🤖 RAG Document Q&A (EN/RO)</p>', unsafe_allow_html=True)

# Load translation models
@st.cache_resource
def load_translation_models():
    ro_en_tokenizer = MarianTokenizer.from_pretrained("BlackKakapo/opus-mt-ro-en")
    ro_en_model = MarianMTModel.from_pretrained("BlackKakapo/opus-mt-ro-en")
    en_ro_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ro")
    en_ro_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ro")
    return ro_en_tokenizer, ro_en_model, en_ro_tokenizer, en_ro_model

ro_en_tokenizer, ro_en_model, en_ro_tokenizer, en_ro_model = load_translation_models()

def translate(text, tokenizer, model):
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", truncation=True)
    translation = model.generate(**tokens)
    return tokenizer.decode(translation[0], skip_special_tokens=True)

# Sidebar for PDF upload
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()

        if not docs:
            st.sidebar.error("❌ Unable to extract content from the PDF.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        if not splits:
            st.sidebar.error("❌ Failed to split document.")
            st.stop()

        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)

        st.sidebar.success(f"✅ PDF processed! Pages: {len(docs)}")

        query = st.text_input("Ask a question (in English or Romanian):", placeholder="Type your question...")

        if query:
            try:
                lang = detect(query)
            except:
                lang = "unknown"

            if lang == "ro":
                with st.spinner("🔄 Translating Romanian → English..."):
                    english_query = translate(query, ro_en_tokenizer, ro_en_model)
            else:
                english_query = query

            results = vectorstore.similarity_search(english_query, k=4)

            if not results:
                st.warning("❓ No relevant results found.")
            else:
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    huggingfacehub_api_token=hf_api_key,
                    max_new_tokens=512
                )
                chat = ChatHuggingFace(llm=llm)
                prompt = pull("rlm/rag-prompt")

                rag_chain = (
                    {
                        "context": vectorstore.as_retriever() | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                        "question": RunnablePassthrough()
                    }
                    | prompt
                    | chat
                    | StrOutputParser()
                )

                try:
                    with st.spinner("💬 Generating answer..."):
                        english_answer = rag_chain.invoke(english_query)

                    if lang == "ro":
                        with st.spinner("🔄 Translating English → Romanian..."):
                            final_answer = translate(english_answer, en_ro_tokenizer, en_ro_model)
                        st.markdown(f'<div class="response"><strong>Răspuns:</strong><br>{final_answer}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="response"><strong>Answer:</strong><br>{english_answer}</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"⚠️ Error generating the response: {e}")

    except Exception as e:
        st.sidebar.error(f"❌ Error processing PDF: {e}")
else:
    st.sidebar.info("Please upload a PDF to get started.")
