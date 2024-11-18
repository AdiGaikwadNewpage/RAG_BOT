from flask import Flask, request, render_template, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
import os
import json


app = Flask(__name__)
app = Flask(__name__, static_folder="static")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

VECTOR_STORE_FOLDER = "db"
if not os.path.exists(VECTOR_STORE_FOLDER):
    os.makedirs(VECTOR_STORE_FOLDER)

# folder_path = "db"

cached_llm = Ollama(model="llama3")
embedding = FastEmbedEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
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

@app.route("/chatbot_html")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_pdf():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_name = file.filename
    save_path = os.path.join(UPLOAD_FOLDER, file_name)
    file.save(save_path)

    loader = PDFPlumberLoader(save_path)
    docs = loader.load_and_split()

    chunks = text_splitter.split_documents(docs)

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=VECTOR_STORE_FOLDER
    )
    vector_store.persist()

    return jsonify({
        "status": "Successfully uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks)
    })


@app.route("/query", methods=["POST"])
def query():
    query = request.json.get("query")

    print(f"query: {query}")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    print("Loading vector store")
    vector_store = Chroma(persist_directory=VECTOR_STORE_FOLDER, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.1},
    )
    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    #rerank

    result = chain.invoke({"input": query})

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"]}
        )

    formatted_answer = result["answer"]
    response = {
        "answer": formatted_answer,
        "sources": sources
    }
    return jsonify(response)




# @app.route("/ai", methods=["POST"])
# def aiPost():
#     print("Post /ai called")
#     json_content = request.json
#     query = json_content.get("query")

#     print(f"query: {query}")

#     response = cached_llm.invoke(query)

#     print(response)

#     response_answer = {"answer": response}
#     return response_answer


# @app.route("/ask_pdf", methods=["POST"])
# def askPDFPost():
#     print("Post /ask_pdf called")
#     json_content = request.json
#     query = json_content.get("query")

#     print(f"query: {query}")

#     print("Loading vector store")
#     vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

#     print("Creating chain")
#     retriever = vector_store.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={
#             "k": 20,
#             "score_threshold": 0.1,
#         },
#     )

#     document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
#     chain = create_retrieval_chain(retriever, document_chain)

#     result = chain.invoke({"input": query})

#     print(result)

#     sources = []
#     for doc in result["context"]:
#         sources.append(
#             {"source": doc.metadata["source"], "page_content": doc.page_content}
#         )

#     response_answer = {"answer": result["answer"], "sources": sources}
#     return response_answer


# @app.route("/pdf", methods=["POST"])
# def pdfPost():
#     file = request.files["file"]
#     file_name = file.filename
#     save_file = "pdf/" + file_name
#     file.save(save_file)
#     print(f"filename: {file_name}")

#     loader = PDFPlumberLoader(save_file)
#     docs = loader.load_and_split()
#     print(f"docs len={len(docs)}")

#     chunks = text_splitter.split_documents(docs)
#     print(f"chunks len={len(chunks)}")

#     vector_store = Chroma.from_documents(
#         documents=chunks, embedding=embedding, persist_directory=folder_path
#     )

#     vector_store.persist()

#     response = {
#         "status": "Successfully Uploaded",
#         "filename": file_name,
#         "doc_len": len(docs),
#         "chunks": len(chunks),
#     }
#     return response


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
