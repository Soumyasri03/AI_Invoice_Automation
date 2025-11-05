# AI-Powered Invoice Assistant

An intelligent document analysis tool that uses a Large Language Model (LLM) to extract, summarize, and answer questions from PDF invoices.

This application provides a simple interface built with Streamlit to manage and query invoice data. Instead of relying on basic keyword searches, it uses an LLM to understand context and deliver clear, conversational answers.

---

## Key Features

- **PDF Ingestion:** Upload multiple PDF invoices to create a searchable knowledge base using LlamaIndex.
- **Interactive Data Viewer:** View all indexed invoices, including parsed details such as customer name, total amount, item lists, and raw extracted text for debugging.
- **Automated Excel Reporting:** Generate and download Excel summaries of invoices using natural language time periods (e.g., “August 2025”, “last month”).
- **LLM-Powered Q&A:** Ask detailed, natural language questions and receive accurate, human-like responses. The system works across industries, whether in electronics, automotive, agriculture, or others.

---

## How the AI Works

The system uses a two-stage process to analyze and respond to your questions:

1. **Smart Query Generation:**  
   When you ask a question like “How many printers did we buy from Charlie Davis?”, the system sends it to an LLM, which generates multiple relevant search queries such as “quantity of printers from Charlie Davis” or “printer purchases from Davis.”

2. **Answer Generation:**  
   These queries are used to search the indexed invoices. The retrieved data is then combined with your original question and sent back to the LLM to produce a clear, concise answer.

This technique is known as Retrieval-Augmented Generation (RAG), enabling more accurate and flexible answers to complex questions.

---

## Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Data Processing:** Pandas, DateParser  
- **AI and Retrieval System:** LlamaIndex  
- **LLM Integration:** LangChain, Groq API (Llama 3.1)  
- **Embeddings:** HuggingFace Sentence-Transformers (local, no API key required)  
- **Environment Management:** `python-dotenv`

---

## Getting Started

### Prerequisites

- Python 3.11 or later  
- A free Groq API key

---

## Project Architecture

The project is organized with a clean separation of concerns:

- **`app.py` (Frontend):** Manages the Streamlit interface, handles user inputs, and displays results.
- **`invoice_tools.py` (Backend):** Handles all core functionality including PDF ingestion, LlamaIndex management, data parsing, and the LLM-powered Q&A system.



