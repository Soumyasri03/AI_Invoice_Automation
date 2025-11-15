# Invoice Automation App

This project is a Streamlit-based web application designed to automate the extraction, indexing, querying, and report generation . The application utilizes LlamaIndex, LangChain, HuggingFace embeddings, and a Groq LLM to read invoice PDFs, extract detailed information, store the processed data in a vector index, and allow users to query invoices or generate Excel reports based on specific time periods.

The purpose of this project is to simplify manual invoice management and build an automated workflow for analyzing multiple invoices efficiently.

---

## Features

### 1. Upload and Process Invoices
- Upload one or multiple invoice PDFs.
- Automatically extracts:
  - Vendor name and address  
  - Buyer name and address  
  - Invoice number  
  - Invoice date  
  - Item descriptions  
  - Subtotal  
  - Tax  
  - Discount  
  - Total amount  
- Extracted data is converted into structured JSON and indexed for querying.

### 2. Query Invoices Using Natural Language
- Ask questions about any invoice.
- Chat-based system supports follow-up questions using chat history.
- Uses semantic search to retrieve the most relevant invoice entries.
- Answers strictly rely on the indexed invoice information.

### 3. Generate Excel Reports
- Export invoice data to Excel based on:
  - Month (e.g., January 2025)
  - Date range (e.g., 21/03/2025 to 21/06/2025)
  - Natural language (e.g., last month, this month)
  - Single date (e.g., 04/09/2025)
  - Quarter (e.g., Q1 2025)
- Filtered invoice results are saved into an Excel spreadsheet.

---

## Technologies Used

- Python  
- Streamlit  
- LangChain  
- LlamaIndex  
- HuggingFace Embeddings  
- Groq LLM (llama-3.1-8b-instant)  
- Pandas  
- Pydantic  
- DateParser  

---

## Project Structure

