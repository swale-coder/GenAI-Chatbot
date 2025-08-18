import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
import os

# --- STEP 1: Set OpenAI API Key and other setup ---
# It's better to get the key from a secure location.
# For this example, we'll assign it from the hardcoded string
# The correct way is to use st.secrets or environment variables


# --- STEP 2: Create Streamlit Interface ---
st.header("GenAI Chatbot")

with st.sidebar:
    st.title("GenAI Chatbot")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# --- STEP 3: Function to Extract File Text ---
def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# --- STEP 4: Function to Split Text into Chunks ---
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", ".", "!", "?", " "],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# --- STEP 5: Function to Generate Embeddings and Vector Store ---
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

# --- Main Logic ---
# Use st.session_state to store data that needs to be persistent
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing..."):
            # Get PDF text
            raw_text = get_pdf_text(file)

            # Get the text chunks
            text_chunks = get_text_chunks(raw_text)

            # Create and store vector store in session state
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.success("PDF processed successfully! You can now ask questions.")

# --- STEP 6: Accept User Questions and Generate Response ---
user_question = st.text_input("Type your question here:", key="user_input")

if user_question and st.session_state.vector_store is not None:
    # Find Relevant Chunks
    match = st.session_state.vector_store.similarity_search(user_question)

    # Define the Language Model
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo"
    )

    # Create QA Chain & Generate Response
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=match, question=user_question)
    st.write(response)