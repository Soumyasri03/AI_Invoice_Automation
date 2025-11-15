import os
import json
import asyncio
import re
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import dateparser

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain / LlamaIndex
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import ConfigurableField
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

from llama_index.core import (
	SimpleDirectoryReader,
	Document,
	StorageContext,
	VectorStoreIndex,
	load_index_from_storage,
	Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ------------------------------
# Config
# ------------------------------
load_dotenv()
st.set_page_config(page_title="Invoice Automation", page_icon="📄", layout="wide")
DEFAULT_INDEX_PATH = "./invoice_index"
DEFAULT_INBOX_DIR = "./invoices_to_process"
DEFAULT_GROQ_API_KEY = ""

# Lazy initialization of embedding model to avoid PyTorch meta tensor issues
_embed_model = None
_embed_model_initialized = False

def _get_embed_model():
	global _embed_model, _embed_model_initialized
	if not _embed_model_initialized:
		_embed_model_initialized = True
		try:
			# Set environment variable to avoid meta tensor issues
			import os
			os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
			
			# Initialize embedding model
			_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
			Settings.embed_model = _embed_model
		except NotImplementedError as e:
			# Handle meta tensor error specifically
			if "meta tensor" in str(e).lower():
				try:
					# Try with trust_remote_code flag if available
					import os
					os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
					_embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
					Settings.embed_model = _embed_model
				except Exception:
					st.warning("Embedding model initialization failed. The app will work but may be slower.")
					Settings.embed_model = None
			else:
				raise
		except Exception as e:
			# Last resort: let LlamaIndex use its default
			st.warning(f"Could not load custom embedding model: {e}. Using default.")
			Settings.embed_model = None
	return _embed_model


_SECRETS_LOCATIONS = [
	Path.home() / ".streamlit" / "secrets.toml",
	Path.cwd() / ".streamlit" / "secrets.toml",
]
HAS_STREAMLIT_SECRETS = any(path.exists() for path in _SECRETS_LOCATIONS)


def _get_secret(key: str, default: str = "") -> str:
	if not HAS_STREAMLIT_SECRETS:
		return default
	try:
		return st.secrets[key]
	except KeyError:
		return default


GROQ_API_KEY = os.getenv("GROQ_API_KEY") or _get_secret("GROQ_API_KEY", DEFAULT_GROQ_API_KEY)
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME") or _get_secret("GROQ_MODEL_NAME", "llama-3.1-8b-instant")


# ------------------------------
# Models
# ------------------------------
class InvoiceDetails(BaseModel):
	vendor_name: str = Field(description="The name of the vendor or company.")
	vendor_address: str = Field(description="Address of the vendor or company.")
	buyer_name: str = Field(description="Name of the buyer or client.")
	buyer_address: str = Field(description="Address of the buyer or client.")
	invoice_number: str = Field(description="The unique invoice identifier.")
	invoice_date: str = Field(description="The date of the invoice (YYYY-MM-DD).")
	items: list[str] = Field(description="List of purchased items with details.")
	subtotal: float = Field(description="Subtotal before tax and discount.")
	tax: float = Field(description="Tax applied to the invoice.")
	discount: float = Field(description="Discount applied to the invoice.")
	total_amount: float = Field(description="The final total amount due after tax and discount.")


# ------------------------------
# LLM Factory (Groq)
# ------------------------------
def get_llm():
	if not GROQ_API_KEY:
		raise ValueError("Missing GROQ API key. Set GROQ_API_KEY in your environment or Streamlit secrets.")
	return ChatGroq(
		groq_api_key=GROQ_API_KEY,
		model_name=GROQ_MODEL_NAME,
		temperature=0
	).configurable_fields(
		callbacks=ConfigurableField(
			id='callbacks',
			name='callbacks',
			description='A list of callbacks to use for streaming'
		)
	)


# ------------------------------
# Core operations
# ------------------------------
async def extract_invoice_details_from_pdf(file_path: str, llm: ChatGroq) -> InvoiceDetails:
	parser = PydanticOutputParser(pydantic_object=InvoiceDetails)
	prompt = PromptTemplate(
		template=(
			"Extract key information from this invoice.\n"
			"{format_instructions}\nDocument:\n{document_text}"
		),
		input_variables=['document_text'],
		partial_variables={'format_instructions': parser.get_format_instructions()},
	)
	chain = prompt | llm | parser

	reader = SimpleDirectoryReader(input_files=[file_path])
	docs = reader.load_data()
	document_text = "\n".join([doc.text for doc in docs])
	result: InvoiceDetails = await chain.ainvoke({'document_text': document_text})
	return result


async def load_invoices_to_index(directory: str, index_path: str, llm: ChatGroq) -> str:
	try:
		invoice_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
		if not invoice_files:
			return f"No PDF files found in '{directory}'."
	except FileNotFoundError:
		return f"Error: The directory '{directory}' does not exist."

	# Build a fresh in-memory storage, then persist after indexing to avoid
	# attempts to read a non-existent persisted index on first run.
	storage_context = StorageContext.from_defaults()
	documents_to_index = []
	processed = 0

	for name in invoice_files:
		file_path = os.path.join(directory, name)
		try:
			inv = await extract_invoice_details_from_pdf(file_path, llm)
			doc = Document(text=json.dumps(inv.model_dump(), indent=2), metadata={'file_name': name})
			documents_to_index.append(doc)
			processed += 1
		except Exception as e:
			msg = str(e)
			# Surface invalid key once and abort further processing to avoid noisy repeats
			if "invalid api key" in msg.lower():
				return "Error: Invalid GROQ API Key. Update the GROQ_API_KEY environment variable and retry."
			st.warning(f"Failed to process {name}: {e}")

	if documents_to_index:
		# Initialize embedding model before creating index
		_get_embed_model()
		index = VectorStoreIndex.from_documents(documents_to_index, storage_context=storage_context)
		os.makedirs(index_path, exist_ok=True)
		index.storage_context.persist(persist_dir=index_path)
	return f"Processed and indexed {processed} of {len(invoice_files)} invoices from '{directory}'."


def load_index(index_path: str):
	storage_context = StorageContext.from_defaults(persist_dir=index_path)
	return load_index_from_storage(storage_context)


def _extract_json_dict(text: str) -> dict:
	if not text:
		return {}
	text = text.strip()
	try:
		return json.loads(text)
	except Exception:
		pass
	match = re.search(r"\{.*\}", text, re.DOTALL)
	if match:
		try:
			return json.loads(match.group(0))
		except Exception:
			return {}
	return {}


def _build_invoice_references(invoices: list[dict]):
	references = []
	id_lookup = {}
	for idx, inv in enumerate(invoices):
		lookup_id = str(inv.get("invoice_number") or f"row-{idx+1}")
		ref = {
			"invoice_id": lookup_id,
			"invoice_number": inv.get("invoice_number"),
			"invoice_date": inv.get("invoice_date"),
			"total_amount": inv.get("total_amount"),
		}
		references.append(ref)
		id_lookup[lookup_id] = idx
	return references, id_lookup


def _select_invoices_for_period(period_text: str, invoice_refs: list[dict]):
	"""
	Select invoices that match a given period using date parsing and comparison.
	Handles natural language dates like "August 2025", date ranges, etc.
	"""
	if not invoice_refs:
		return [], "No invoice metadata available."
	
	# Parse the period text into start and end dates
	def resolve_dates(text: str):
		text = text.strip()
		text_lower = text.lower()
		
		# Month name to number mapping
		month_map = {
			"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
			"july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
		}
		
		# FIRST: Try to match "Month Year" pattern explicitly (e.g., "August 2025")
		# This handles cases where dateparser might not parse correctly
		for month_name, month_num in month_map.items():
			if month_name in text_lower:
				# Extract year from text (4-digit year)
				year_match = re.search(r'\b(19|20)\d{2}\b', text)
				if year_match:
					year = int(year_match.group(0))
					# Return full month range
					first = datetime(year, month_num, 1).date()
					if month_num == 12:
						next_month = datetime(year + 1, 1, 1).date()
					else:
						next_month = datetime(year, month_num + 1, 1).date()
					last = next_month - timedelta(days=1)
					return first, last
		
		# Case: explicit range "… to …" or "… - …"
		if " to " in text_lower or " - " in text_lower:
			separator = " to " if " to " in text_lower else " - "
			parts = text_lower.split(separator, 1)
			if len(parts) == 2:
				left, right = [p.strip() for p in parts]
				d1 = dateparser.parse(left, settings={"DATE_ORDER": "DMY"})
				d2 = dateparser.parse(right, settings={"DATE_ORDER": "DMY"})
				if d1 and d2:
					return d1.date(), d2.date()
		
		# Single date-like phrase (August 2025, 04/09/2025, this week, last month, etc.)
		parsed = dateparser.parse(text, settings={"DATE_ORDER": "DMY"})
		if parsed:
			start = parsed.date()
			end = start
			
			# If phrase implies a week, expand to Monday–Sunday
			if "week" in text.lower():
				monday = start - timedelta(days=start.weekday())
				return monday, monday + timedelta(days=6)
			
			# If phrase implies a month (e.g., "August 2025", "January 2025")
			# Check if text contains month name or if parsed date is the first of a month
			month_keywords = ["january", "february", "march", "april", "may", "june",
			                 "july", "august", "september", "october", "november", "december"]
			has_month_keyword = any(month in text.lower() for month in month_keywords)
			is_first_of_month = start.day == 1
			
			# Also check for quarter patterns like "Q1 2025"
			has_quarter = bool(re.search(r'\bq[1-4]\s*\d{4}\b', text.lower()))
			
			if has_month_keyword or (is_first_of_month and not has_quarter):
				# Expand to full month
				first = start.replace(day=1)
				# Get last day of month
				if first.month == 12:
					next_month = first.replace(year=first.year + 1, month=1)
				else:
					next_month = first.replace(month=first.month + 1)
				last = next_month - timedelta(days=1)
				return first, last
			
			# Handle quarter patterns
			if has_quarter:
				# Extract quarter and year
				quarter_match = re.search(r'\bq([1-4])\s*(\d{4})\b', text.lower())
				if quarter_match:
					quarter = int(quarter_match.group(1))
					year = int(quarter_match.group(2))
					# Calculate quarter date range
					first_month = (quarter - 1) * 3 + 1
					first = datetime(year, first_month, 1).date()
					last_month = quarter * 3
					if last_month == 12:
						next_month = datetime(year + 1, 1, 1).date()
					else:
						next_month = datetime(year, last_month + 1, 1).date()
					last = next_month - timedelta(days=1)
					return first, last
			
			return start, end
		
		return None, None
	
	start_date, end_date = resolve_dates(period_text)
	if not start_date:
		return [], f"Could not interpret time period '{period_text}'. Please use formats like 'August 2025', '2025-08-01 to 2025-08-31', etc."
	
	# Parse invoice dates and match them
	def parse_inv_date(value):
		if not value:
			return None
		try:
			# Try YYYY-MM-DD format first
			return datetime.strptime(str(value), "%Y-%m-%d").date()
		except Exception:
			try:
				# Try other common formats
				return dateparser.parse(str(value)).date() if dateparser.parse(str(value)) else None
			except Exception:
				return None
	
	selected_ids = []
	debug_info = []
	for ref in invoice_refs:
		inv_date_str = ref.get("invoice_date", "")
		inv_date = parse_inv_date(inv_date_str)
		if inv_date:
			if start_date <= inv_date <= end_date:
				selected_ids.append(ref["invoice_id"])
			# Collect debug info for first few invoices
			if len(debug_info) < 5:
				debug_info.append(f"Date: {inv_date_str} -> {inv_date} (in range: {start_date <= inv_date <= end_date})")
		else:
			if len(debug_info) < 5:
				debug_info.append(f"Date: {inv_date_str} -> Could not parse")
	
	summary = f"Matched {len(selected_ids)} invoice(s) for period '{period_text}' (dates from {start_date} to {end_date})"
	if not selected_ids and debug_info:
		summary += f"\nDebug: {'; '.join(debug_info)}"
	return selected_ids, summary


def query_invoice_data(index_path: str, question: str, chat_history: list = None) -> str:
	"""
	Query invoice data with conversation memory.
	
	Args:
		index_path: Path to the invoice index
		question: Current user question
		chat_history: List of previous messages (HumanMessage, AIMessage) for context
	"""
	if chat_history is None:
		chat_history = []
	
	try:
		index = load_index(index_path)
	except Exception:
		return "Error: The invoice index does not exist. Please process invoices first."
	retriever = index.as_retriever(similarity_top_k=5)
	nodes = retriever.retrieve(question)
	if not nodes:
		return "No relevant information found for that question."

	context_parts = []
	for node in nodes:
		meta = node.metadata or {}
		label = meta.get("file_name", "Invoice document")
		context_parts.append(f"Source: {label}\n{node.get_content()}")
	context = "\n\n---\n\n".join(context_parts)

	# Build conversation history context
	history_context = ""
	if chat_history:
		history_context = "\n\nPrevious conversation:\n"
		for msg in chat_history[-6:]:  # Include last 3 exchanges (6 messages)
			if isinstance(msg, HumanMessage):
				history_context += f"User: {msg.content}\n"
			elif isinstance(msg, AIMessage):
				history_context += f"Assistant: {msg.content}\n"

	prompt = (
		"You are an assistant that analyzes structured invoice data in JSON format.\n"
		"Answer the user's question using only the provided invoices. "
		"If the answer is not present, say you do not know.\n"
		"You can use the previous conversation context to understand follow-up questions.\n\n"
		f"Invoice entries:\n{context}\n"
		f"{history_context}\n"
		f"Current question: {question}\nAnswer:"
	)
	llm = get_llm()
	response = llm.invoke(prompt)
	return getattr(response, "content", str(response))


def create_invoice_excel_report(index_path: str, period_text: str, output_path: str) -> str:
	try:
		index = load_index(index_path)
	except Exception:
		return "Error: The invoice index does not exist. Please process invoices first."

	retriever = index.as_retriever(similarity_top_k=1000)
	nodes = retriever.retrieve("all invoices")
	all_invoice_data = []
	for node in nodes:
		payload = _extract_json_dict(node.get_content())
		if payload:
			all_invoice_data.append(payload)

	if not all_invoice_data:
		return "No invoices available to export. Please process invoices first."

	invoice_refs, id_lookup = _build_invoice_references(all_invoice_data)
	selected_ids, summary = _select_invoices_for_period(period_text, invoice_refs)

	if not selected_ids:
		# Provide more detailed error message with debugging info
		error_msg = f"Could not match any invoices to '{period_text}'."
		if summary:
			error_msg += f"\n\n{summary}"
		return error_msg

	filtered = [all_invoice_data[id_lookup[iid]] for iid in selected_ids if iid in id_lookup]

	if not filtered:
		return f"The LLM did not select any invoices for '{period_text}'."

	try:
		pd.DataFrame(filtered).to_excel(output_path, index=False, engine="openpyxl")
		reason = f" Reason: {summary}" if summary else ""
		return f"Excel report with {len(filtered)} invoices created for '{period_text}' at '{output_path}'.{reason}"
	except Exception as e:
		return f"Failed to create the Excel report: {e}"


# ------------------------------
# UI
# ------------------------------
st.title("📄 Invoice Automation App")

with st.sidebar:
	st.header("Settings")
	if not GROQ_API_KEY:
		st.error("Missing GROQ_API_KEY environment variable. Set it before running actions.")
	index_path = st.text_input("Index directory", value=DEFAULT_INDEX_PATH)
	inbox_dir = st.text_input("Invoices folder", value=DEFAULT_INBOX_DIR)
	col_sb1, col_sb2 = st.columns(2)
	with col_sb1:
		if st.button("Create folders", use_container_width=True):
			os.makedirs(index_path, exist_ok=True)
			os.makedirs(inbox_dir, exist_ok=True)
			st.success("Folders ensured.")
	with col_sb2:
		if st.button("Clear index", type="secondary", use_container_width=True):
			try:
				import shutil
				shutil.rmtree(index_path)
				st.success("Index cleared.")
			except Exception as e:
				st.warning(f"Could not clear index: {e}")

tab1, tab2, tab3 = st.tabs(["Upload & Index", "Query", "Export Report"])

with tab1:
	st.subheader("Upload PDFs and Build Index")
	uploaded_files = st.file_uploader("Upload invoice PDFs", type=["pdf"], accept_multiple_files=True)
	if uploaded_files:
		os.makedirs(inbox_dir, exist_ok=True)
		for up in uploaded_files:
			dest = os.path.join(inbox_dir, up.name)
			with open(dest, "wb") as f:
				f.write(up.read())
		st.success(f"Saved {len(uploaded_files)} file(s) to '{inbox_dir}'.")

	if st.button("Process & Index Invoices", type="primary"):
		if not GROQ_API_KEY:
			st.error("Missing GROQ_API_KEY environment variable. Set it and restart the app.")
		else:
			llm = get_llm()
			with st.spinner("Processing invoices..."):
				msg = asyncio.run(load_invoices_to_index(inbox_dir, index_path, llm))
			st.success(msg)

with tab2:
	st.subheader("Ask Questions about Invoices")
	
	# Initialize chat history in session state
	if "chat_history" not in st.session_state:
		st.session_state.chat_history = []
	
	# Display conversation history
	for msg in st.session_state.chat_history:
		if isinstance(msg, HumanMessage):
			with st.chat_message("user"):
				st.write(msg.content)
		elif isinstance(msg, AIMessage):
			with st.chat_message("assistant"):
				st.write(msg.content)
	
	# Input for new question
	question = st.chat_input("Ask a question about your invoices...")
	
	# Handle question submission
	if question:
		# Add user message to history
		st.session_state.chat_history.append(HumanMessage(content=question))
		
		# Get answer with conversation context
		with st.spinner("Querying index..."):
			answer = query_invoice_data(index_path, question, st.session_state.chat_history)
		
		# Add assistant response to history
		st.session_state.chat_history.append(AIMessage(content=answer))
		
		# Rerun to update the display with new messages
		st.rerun()
	
	# Clear conversation button
	if st.button("Clear Conversation", type="secondary"):
		st.session_state.chat_history = []
		st.rerun()

with tab3:
	st.subheader("Create Excel Report")
	
	with st.expander("📅 Date Format Examples", expanded=False):
		st.markdown("""
		**You can use natural language or specific dates. Examples:**
		- `January 2025` - All invoices from January 2025
		- `21/03/2025 to 21/06/2025` - Date range (DD/MM/YYYY format)
		- `2025-03-21 to 2025-06-21` - Date range (YYYY-MM-DD format)
		- `last month` - Previous month
		- `this month` - Current month
		- `Q1 2025` - First quarter of 2025
		- `March 2025` - All invoices from March 2025
		- `04/09/2025` - Single date (invoices on that date)
		- `before 15/06/2025` - All invoices before this date
		- `after 01/01/2025` - All invoices after this date
		- `between 01/03/2025 and 31/03/2025` - Date range
		""")
	
	col_a, col_b = st.columns([2, 1])
	with col_a:
		period_text = st.text_input("Period", placeholder="e.g., 'January 2025' or '21/03/2025 to 21/06/2025'", help="Enter a date range or period description")
	with col_b:
		default_out = os.path.join(os.getcwd(), "invoice_report.xlsx")
		output_path = st.text_input("Output path", value=default_out)

	if st.button("Generate Excel", type="primary"):
		if not period_text.strip():
			st.warning("Provide a period.")
		else:
			with st.spinner("Generating report..."):
				msg = create_invoice_excel_report(index_path, period_text, output_path)
			st.write(msg)


# Footer
st.caption("Built for the uploaded notebook workflow: upload → index → query → export.")


