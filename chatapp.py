import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Paths
# Profile Image URL
IMAGE_URL = "https://media.licdn.com/dms/image/v2/D5603AQEyiwmi_N1P1w/profile-displayphoto-shrink_800_800/B56ZVM.D5yHEAc-/0/1740753096373?e=1785974400&v=beta&t=QCr_aqcp7oxegh1uBXvt4MpkBwwkxiH1M-B4-ZJqZ3A"

# Professional Prussian Blue & Tropical Teal Theme
CUSTOM_THEME_CSS = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Global Background - Prussian Blue */
    .stApp {
        background-color: #0b132b !important; 
        color: #e2e8f0 !important;
    }
    
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    
    /* Left Sidebar - Space Indigo */
    section[data-testid="stSidebar"] {
        background-color: #1c2541 !important;
        border-right: 1px solid #3a506b !important;
    }
    
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label {
        color: #6fffe9 !important;
    }
    
    /* Document Upload Card */
    .upload-card {
        background: #0b132b;
        border: 1px solid #3a506b;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 20px;
    }
    
    /* Profile Image Styling */
    [data-testid="stImage"] img {
        border-radius: 12px !important;
        border: 2px solid #5bc0be !important;
        box-shadow: 0 4px 15px rgba(91, 192, 190, 0.2);
    }
    
    /* Primary Button - Tropical Teal */
    .stButton>button {
        background: linear-gradient(135deg, #3a506b 0%, #5bc0be 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #5bc0be 0%, #6fffe9 100%) !important;
        color: #0b132b !important;
        box-shadow: 0 4px 15px rgba(111, 255, 233, 0.4) !important;
    }
    
    /* Chat Input Target */
    div[data-testid="stChatInput"] {
        background-color: #1c2541 !important;
        border: 1px solid #3a506b !important;
    }
</style>
"""

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are Divyansh's AI Assistant, an expert data and operations copilot.
    
    Follow these exact instructions to answer the user's query:
    1. DOCUMENT QUERY: Look at the provided 'Context'. If the answer to the user's question is found there, provide a highly detailed response based ONLY on that context.
    2. GENERAL KNOWLEDGE QUERY: If the user asks a general knowledge question, a casual greeting, or asks about universal facts (e.g., "What is the capital of Mexico?", "How is the weather in Bangalore?", etc.), answer it normally using your general intelligence.
    3. MISSING DATA: If the user explicitly asks about the uploaded documents, but the information is missing from the 'Context', politely state: "That information is not available in the uploaded documents."
    
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def query_rag_engine(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    if not os.path.exists("faiss_index"):
        return "System Warning: Database is empty. Please upload and process a document in the sidebar first."
        
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def main():
    st.set_page_config(page_title="Divyansh's Assistant", page_icon="📊", layout="wide")
    st.markdown(CUSTOM_THEME_CSS, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Left Column: Minimalist App Information & Data Upload Panel
    with st.sidebar:
        st.markdown(
"""
<div style="padding-top: 10px; margin-bottom: 20px;">
    <h3 style="margin: 0; color:#6fffe9; font-size: 1.2rem; font-weight: 700; letter-spacing: 0.5px;">
        <i class="fa-solid fa-server" style="color: #5bc0be; margin-right: 8px;"></i>Divyansh's Assistant
    </h3>
    <p style="margin: 5px 0 12px 0; color: #5bc0be !important; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">
        Multi-PDF AI Agent
    </p>
    <p style="color: #e2e8f0; font-size: 0.8rem; line-height: 1.5; margin: 0; background: #0b132b; border: 1px solid #3a506b; border-radius: 6px; padding: 10px;">
        Upload operational PDFs to instantly extract actionable insights using advanced semantic search.
    </p>
</div>
<hr style="border-color: #3a506b; margin-bottom: 20px;">
""", 
        unsafe_allow_html=True)
        
        st.markdown(
"""
<div class="upload-card">
    <p style="font-size: 0.85rem; font-weight: 600; margin-bottom: 5px; color:#5bc0be;">
        <i class="fa-solid fa-file-pdf" style="margin-right: 8px;"></i> Target Documentation
    </p>
</div>
""", 
        unsafe_allow_html=True)
        
        pdf_docs = st.file_uploader(
            "Upload files:", 
            accept_multiple_files=True,
            type=["pdf"],
            label_visibility="collapsed"
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Indexing vector database..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.toast("Documents processed successfully!", icon="✅")
            else:
                st.warning("Please upload a PDF first.")
                
        # Sidebar Footer
        st.markdown(
"""
<div style="margin-top: 40px; padding: 15px; text-align: center; color: #5bc0be; font-size: 0.75rem; font-weight: 600; border-top: 1px solid #3a506b;">
    <i class="fa-solid fa-chart-line" style="margin-right: 6px;"></i> Created by Divyansh<br>Associate Business Analyst @ Swiggy
</div>
""", 
        unsafe_allow_html=True)

    # Right Column: Main Profile & Bio (Condensed and Professional)
    with st.container():
        img_col, text_col = st.columns([1, 7])
        
        with img_col:
            if os.path.exists(IMAGE_URL):
                st.image(IMAGE_URL, use_container_width=True)
            else:
                st.markdown('<div style="text-align: center; padding-top:5px;"><i class="fa-solid fa-user-tie" style="font-size: 3rem; color: #5bc0be;"></i></div>', unsafe_allow_html=True)
            
        with text_col:
            st.markdown(
"""
<div style="padding-left: 10px;">
    <h1 style="margin: 0; font-size: 1.6rem; color: #6fffe9; display: flex; align-items: center;">
        <i class="fa-solid fa-layer-group" style="color: #5bc0be; margin-right: 12px; font-size: 1.4rem;"></i>
        Multi-PDF AI Agent
    </h1>
    <p style="margin: 8px 0 0 0; color: #e2e8f0; font-size: 0.9rem; line-height: 1.5;">
        I am <strong>Divyansh Mishra</strong>, an Associate Business Analyst at Swiggy specializing in e-commerce analytics, ETL pipelines, and data visualization. I engineered this localized Retrieval-Augmented Generation (RAG) architecture using LangChain, FAISS, and Google Gemini. By bridging data engineering with warehouse execution, this tool enables cross-functional teams to instantly query dense standard operating procedures and extract highly accurate, hallucination-free insights.
    </p>
</div>
""", 
            unsafe_allow_html=True)

    st.write("---")

    # Chat Interaction Interface
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
            <div style="display: flex; justify-content: flex-end; margin-bottom: 20px;">
                <div style="background: #3a506b; padding: 12px 18px; border-radius: 12px 12px 0px 12px; color: #ffffff; max-width: 75%; border: 1px solid #5bc0be;">
                    <div style="font-size: 0.7rem; color: #6fffe9; font-weight: 600; margin-bottom: 4px; text-transform: uppercase;">
                        <i class="fa-solid fa-user" style="margin-right: 5px;"></i> You
                    </div>
                    <div style="font-size: 0.9rem;">{message["content"]}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div style="display: flex; justify-content: flex-start; margin-bottom: 20px;">
                <div style="background: #1c2541; border: 1px solid #3a506b; padding: 16px; border-radius: 12px 12px 12px 0px; color: #e2e8f0; max-width: 80%; border-left: 4px solid #5bc0be;">
                    <div style="font-size: 0.7rem; color: #5bc0be; font-weight: 600; margin-bottom: 8px; text-transform: uppercase;">
                        <i class="fa-solid fa-robot" style="margin-right: 6px;"></i> Assistant
                    </div>
                    <div style="line-height: 1.5; font-size: 0.9rem;">{message["content"]}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    # Chat Input Hook
    if user_question := st.chat_input("Query your operational documents or ask a general question..."):
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            answer = query_rag_engine(user_question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

if __name__ == "__main__":
    main()