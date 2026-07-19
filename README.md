# 📊 Operations RAG Agent: Enterprise Multi-PDF AI Copilot

An intelligent Retrieval-Augmented Generation (RAG) agent engineered to seamlessly process complex operational documentation and extract accurate, hallucination-free insights. Built using LangChain, FAISS Vector DB, and Google Gemini Pro, this assistant serves as an interactive data copilot for business operations.

## 📢 Live Demo
👉 **[Launch App On Streamlit Cloud](https://share.streamlit.io/)** *(Update this link with your actual active deployment URL)*

---

## 🏗️ Technical Architecture & How It Works

The architecture transforms unstructured PDF text data into structured, semantically searchable operational intelligence through a multi-stage pipeline:

1. **Document Ingestion:** Reads multi-document batches using a robust `PdfReader` layer.
2. **Dynamic Semantic Chunking:** Splits dense text blocks into clean, uniform paragraphs via `RecursiveCharacterTextSplitter` (configured at 1000 character chunks with 200 character overlaps) to maintain context boundaries.
3. **Vector Embeddings Mapping:** Converts processed strings into mathematical coordinate arrays using `GoogleGenerativeAIEmbeddings` (`models/gemini-embedding-001`).
4. **Local Database Indexing:** Structures text chunks into a memory-efficient `FAISS` vector database index, utilizing local serialization for maximum execution speed.
5. **Contextual Retrieval Search:** Performs semantic similarity matching against user queries, filtering only relevant source paragraphs.
6. **Hallucination-Proof Synthesis:** Feeds retrieved context matrices into `gemini-2.5-flash` with a fine-tuned creativity control parameter (`temperature=0.5`) and custom hybrid system prompt constraints.

---

## 🎯 Key Features

- **Multi-Document Analytical Range:** Resolves semantic cross-references across multiple business files simultaneously.
- **Precision Temperature Guardrails:** System logic prevents AI guesswork—if data isn't explicitly located within the context coordinates, it gracefully returns an empty token alternative flag.
- **Customized Enterprise UI:** Re-skinned with a clean, dual-tone Prussian Blue and Tropical Teal dashboard layout designed for high scannability.
- **Optimized Python Processing:** Assembled purely within Python without external layout overhead, ensuring lightweight cloud deployment.

---

## 🌟 Technical Prerequisites & Libraries

- **Streamlit** - Interface creation dashboard tool.
- **langchain** - Application control middleware routing data flow paths.
- **faiss-cpu** - Facebook AI Similarity Search engine for vector neighborhood sorting.
- **langchain-google-genai** - Interface layer binding LangChain parameters to Google Gemini foundational models.
- **PyPDF2** - Clean text content extraction wrapper.
- **python-dotenv** - Application security isolation container for private credentials.

---

## ▶️ Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/divyansh26-pbox/Multi-PDF-AI-Agent-using-RAG.git](https://github.com/divyansh26-pbox/Multi-PDF-AI-Agent-using-RAG.git)