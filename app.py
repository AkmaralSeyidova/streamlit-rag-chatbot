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
import os
from PIL import Image

# Load environment variables
load_dotenv()
hf_api_key = os.getenv("HF_API_KEY")

# API key validation
if not hf_api_key:
    st.error("❌ Hugging Face API key is missing. Please configure it in the .env file.")
    st.stop()

# Streamlit app setup
st.set_page_config(page_title="RAG with LangChain", layout="centered")

# Custom styling
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

# Title
st.markdown('<p class="title">🤖 RAG with LangChain & HuggingFace</p>', unsafe_allow_html=True)

# Sidebar for PDF upload
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    try:
        import tempfile

        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the PDF file
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()


        if not docs:
            st.sidebar.error("❌ Unable to extract content from the PDF.")
            st.stop()

        # Text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        if not splits:
            st.sidebar.error("❌ Failed to split document.")
            st.stop()

        # Embeddings and vectorstore
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(splits, embedding=embeddings)

        st.sidebar.success(f"✅ PDF processed! Pages: {len(docs)}")

        # User query input
        query = st.text_input("Ask a question about the document:", placeholder="Type your question here...")

        if query:
            # Similarity search
            results = vectorstore.similarity_search(query, k=4)

            if not results:
                st.warning("❓ No relevant results found.")
            else:
                # Set up Hugging Face LLM
                llm = HuggingFaceEndpoint(
                    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                    huggingfacehub_api_token=hf_api_key,
                    max_new_tokens=512
                )
                chat = ChatHuggingFace(llm=llm)

                # Load prompt and create RAG chain
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
                    response = rag_chain.invoke(query)
                    st.markdown(f'<div class="response"><strong>Answer:</strong><br>{response}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"⚠️ Error generating the response: {e}")

    except Exception as e:
        st.sidebar.error(f"❌ Error processing PDF: {e}")
else:
    st.sidebar.info("Please upload a PDF to get started.")
