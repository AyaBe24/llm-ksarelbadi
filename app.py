from flask import Flask, request, jsonify
import os
from langchain_community.llms import Ollama
from langchain_chroma import Chroma  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


embed = FastEmbedEmbeddings()

app = Flask(__name__)

folder_path = "db"
# Create the pdf directory if it does not exist
pdf_directory = "pdf"
if not os.path.exists(pdf_directory):
    os.makedirs(pdf_directory)

cached_llm = Ollama(model="llama3")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)
raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

# Endpoint to receive JSON data
@app.route('/ai', methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    if not query:
        return jsonify({"error": "Query not provided"}), 400

    print(f"query: {query}")

    response = cached_llm.invoke(query)
    print(response)

    response_answer = {"answer": response}
    return jsonify(response_answer)

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embed)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route('/pdf', methods=["POST"])
def pdfPost():
    # Check if file exists in request
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    # Check if file is empty
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_name = file.filename

    # Construct the save path for the file
    save_file = os.path.join(pdf_directory, file_name)

    # Try saving the file
    try:
        file.save(save_file)
        print(f"filename: {file_name}")

        # Logic to process the PDF using the saved file path
        loader = PDFPlumberLoader(save_file)
        docs = loader.load_and_split()
        print(f"docs len={len(docs)}")
        chunks = text_splitter.split_documents(docs)
        print(f"chunks len={len(chunks)}")

        # Updated usage of Chroma
        vector_store = Chroma.from_documents(documents=chunks, embedding=embed, persist_directory=folder_path)
        

        response = {"status": "Successfully Uploaded", "filename": file_name, "document_len": len(docs), "chunks": len(chunks)}
        return jsonify(response)
    except Exception as e:
        print(f"Error saving file: {e}")
        return jsonify({"error": "Failed to save file"}), 500

def start_app():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    start_app()
