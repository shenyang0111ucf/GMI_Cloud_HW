from langchain.llms.base import LLM
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional, List
import requests
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.documents import Document

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

import tkinter as tk
from tkinter import filedialog, messagebox
import fitz  # PyMuPDF
import os
from io import BytesIO

import hashlib
import pickle
import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory
from fastapi.middleware.cors import CORSMiddleware
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Ijk5ZTMyMjI4LTk1YWMtNDdjZS1iMTc1LWI4YmJkNWZlMDYyYSIsInR5cGUiOiJpZV9tb2RlbCJ9.wf85b2eocjJvlm4PFePDfkpnjpJ0qmj2jumvAj3XGk4"

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        image_url, pdf_url = get_pdf_first_page_image(file)
        return jsonify({
            "image_url": image_url,
            "pdf_url": pdf_url
        })

    return jsonify({"error": "Unknown error"}), 500

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

class DeepSeekAPI(LLM):
    api_url: str
    api_key: str

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 1
        }
        # response_test = requests.get("https://api.gmi-serving.com/v1/models", headers=headers)
        # result_test=response_test.json()
        # model_names = result_test.get("text") or result_test.get("data") or str(result_test)

        # print("Available Models", model_names)


        response = requests.post(self.api_url, headers=headers, json=payload)
        print("Status Code:", response.status_code)

        # Raise an error if the response isn't successful
        response.raise_for_status()

        try:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except ValueError:
            raise ValueError("Failed to decode JSON. Raw response: " + response.text)

    @property
    def _identifying_params(self):
        return {"api_url": self.api_url}

    @property
    def _llm_type(self):
        return "GMI_DeepSeek-R1_API"


def get_pdf_first_page_image(file):
    temp_pdf_path = os.path.join(UPLOAD_FOLDER, "temp.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(file.read())

    doc = fitz.open(temp_pdf_path)
    pix = doc[0].get_pixmap()
    image_path = os.path.join(UPLOAD_FOLDER, "first_page.png")
    pix.save(image_path)
    temp_pdf_url="/" + UPLOAD_FOLDER + "/temp.pdf"
    image_url = "/" + UPLOAD_FOLDER + "/first_page.png"

    return image_url, temp_pdf_url

def upload_pdf(file_path):
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6Ijk5ZTMyMjI4LTk1YWMtNDdjZS1iMTc1LWI4YmJkNWZlMDYyYSIsInR5cGUiOiJpZV9tb2RlbCJ9.wf85b2eocjJvlm4PFePDfkpnjpJ0qmj2jumvAj3XGk4"
    # Select a file (returns full path)
    # file_path = filedialog.askopenfilename(
    #     title="Select PDF File",
    #     filetypes=[("PDF files", "*.pdf")]
    # )
    if file_path:
        #try:
        #root.destroy()
        # Open and read the file as bytes, then wrap in BytesIO
        with open(file_path, "rb") as f:
            file_like = BytesIO(f.read())  # Now like a web upload object
            print(file_like.filename)
            image_path, pdf_path  = get_pdf_first_page_image(file_like)
            # messagebox.showinfo("Success", f"Image saved at: {image_path}")

        # documents = get_pdf_object(file_like)
        # summary = get_summarization(documents=documents, api_key=api_key)
        # print(summary)
        # response = question_answering(documents)
        # print(response)

        # except Exception as e:
        #     messagebox.showerror("Error", f"Failed to process PDF: {e}")
    # else:
    #     messagebox.showwarning("No File", "No PDF file selected!")

# def question_answering(file):
#     documents = get_pdf_object(file)
#     my_retriever = create_retriever(documents)
#     llm = DeepSeekAPI(api_url="https://api.gmi-serving.com/v1/chat/completions", api_key=API_KEY)
#     qa_chain = build_qa_chain(my_retriever, llm)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
#     docs = splitter.split_documents(documents)
#     user_input = "Why you use cycle consistency?"
#     response = qa_chain.invoke({"input": user_input, "context": docs})['answer']
#     return response

@app.route('/question_answering', methods=['POST'])
def question_answering_api():
    if 'file' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing file or question'}), 400

    file = request.files['file']
    user_input = request.form['question']

    try:
        # Load PDF and create retriever
        documents = get_pdf_object(file)
        my_retriever = create_retriever(documents)
        llm = DeepSeekAPI(api_url="https://api.gmi-serving.com/v1/chat/completions", api_key=API_KEY)
        qa_chain = build_qa_chain(my_retriever, llm)

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
        docs = splitter.split_documents(documents)

        # Ask the question
        result = qa_chain.invoke({"input": user_input, "context": docs})
        response = result['answer'] if 'answer' in result else result

        return jsonify({'answer': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_pdf_object(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getvalue())
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    return documents


@app.route('/summarize_pdf', methods=['POST'])
def summarize_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        documents = get_pdf_object(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse PDF: {str(e)}"}), 500

    llm = DeepSeekAPI(api_url="https://api.gmi-serving.com/v1/chat/completions", api_key=API_KEY)

    full_text = "\n".join([doc.page_content for doc in documents])

    splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
    docs = splitter.split_text(full_text)

    summary_prompt = PromptTemplate.from_template("Summarize the following text:\n\n{context}")
    chain = summary_prompt | llm | StrOutputParser()

    try:
        summaries = [chain.invoke({"context": doc}) for doc in docs]

        if len(summaries) == 1:
            final_summary = summaries[0]
        else:
            full_summary = "\n".join(summaries)
            final_summary = chain.invoke({"context": full_summary})

        return jsonify({"summary": final_summary})

    except Exception as e:
        return jsonify({"error": f"Summarization failed: {str(e)}"}), 500



# Need to spend some time to check if this one is 100% correct.
def create_retriever(documents, vector_store_path="vector_store"):
    # Generate a hash from the document content to use as a unique ID
    doc_hash = hashlib.md5("".join([doc.page_content for doc in documents]).encode()).hexdigest()
    faiss_path = os.path.join(vector_store_path, f"{doc_hash}_index")
    config_path = os.path.join(vector_store_path, f"{doc_hash}_config.pkl")

    # Create directory if needed
    os.makedirs(vector_store_path, exist_ok=True)

    if os.path.exists(faiss_path) and os.path.exists(config_path):
        # Load FAISS vector store
        with open(config_path, "rb") as f:
            embedding = pickle.load(f)
        vector = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    else:
        # Split and embed documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=0)
        docs = splitter.split_documents(documents)

        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vector = FAISS.from_texts(
            [doc.page_content for doc in docs],
            embedder,
            metadatas=[{"source": doc.metadata.get("source", "Uploaded PDF")} for doc in docs]
        )

        # Save FAISS and embedder
        vector.save_local(faiss_path)
        with open(config_path, "wb") as f:
            pickle.dump(embedder, f)
    print("retriever")
    return vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def build_prompt():
    prompt = """
    1. Use the following context to answer the question.
    2. If you don't know the answer, say "I don't know."
    3. Keep the answer concise (3-4 sentences).

    Context: {context}

    Question: {input}

    Helpful Answer:"""
    return PromptTemplate.from_template(prompt)

def build_qa_chain(retriever, llm):
    # Create the prompt template for stuffing
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}"
    )
    # Stuff documents into the LLM with a context prompt
    prompt = build_prompt()
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=document_prompt
    )
    # Build the final Retrieval QA chain
    retrieval_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain,
    )
    return retrieval_chain

if __name__ == '__main__':
    app.run(debug=True)