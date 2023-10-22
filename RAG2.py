import streamlit as st
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO)

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0
    )
    return llm

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    content = ''
    for page in data:
        content += page.page_content
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(content)
    documents = [Document(page_content=t) for t in chunks]
    return documents

def llm_pipeline(file_path):
    documents = file_processing(file_path)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    llm_answer_gen = load_llm()
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                          chain_type="stuff", 
                                                          retriever=vector_store.as_retriever())
    return answer_generation_chain

def run_app():
    st.title("Question Answer Generator using Mistral 7B")

    uploaded_file = st.file_uploader("Upload your PDF file here", type=['pdf'])

    if uploaded_file:
        with st.spinner("Analyzing..."):
            # Saving the uploaded file temporarily
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Call your pipeline function
            answer_generation_chain = llm_pipeline("temp_pdf.pdf")

        st.success("PDF Analyzed! You can now ask questions.")

        # Getting question from user
        question = st.text_input("Posez votre question ici")

        if st.button("Ask"):
            with st.spinner("Fetching answer..."):
                response = answer_generation_chain.run(question)
                st.write(response)

run_app()