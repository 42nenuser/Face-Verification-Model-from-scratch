This Python code sets up a Retrieval Augmented Generation (RAG) system using the Ollama language model and the Gradio library for a user interface. It allows users to ask questions about a specific Wikipedia page (Ohiya) and get answers based on the information extracted from that page.

Importing Libraries
 
import gradio as gr
import ollama
from bs4 import BeautifulSoup as bs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
Use code with caution
These lines import the necessary libraries:

gradio: For building the user interface.
ollama: For interacting with the Ollama language model.
BeautifulSoup: For parsing HTML (unused in this code).
RecursiveCharacterTextSplitter: For splitting text into chunks.
WebBaseLoader: For loading data from a website.
Chroma: For creating a vector database.
OllamaEmbeddings: For generating text embeddings.
Loading and Preparing Data
 
url = 'https://en.wikipedia.org/wiki/Ohiya'
loader = WebBaseLoader(url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
Use code with caution
Data Loading: The code loads the content of the Wikipedia page about Ohiya using WebBaseLoader.
Text Splitting: The loaded content is split into smaller chunks using RecursiveCharacterTextSplitter.
Embeddings and Vector Store: Embeddings are generated for each chunk using OllamaEmbeddings, and these embeddings are stored in a vector database (Chroma) for efficient searching.
Defining the Ollama LLM Function
 
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']
Use code with caution
This function takes a question and context as input, formats them into a prompt, sends the prompt to the Ollama Llama3 model, and returns the model's response.

Setting up the RAG Chain
 
retriever = vectorstore.as_retriever()

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)
Use code with caution
This part sets up the Retrieval Augmented Generation (RAG) process:

A retriever is created from the vector database to search for relevant documents.
The rag_chain function takes a question, retrieves relevant documents using the retriever, formats the context from those documents, and calls the ollama_llm function to get the answer.
Creating the Gradio Interface
 
def get_important_facts(question):
    return rag_chain(question)

iface = gr.Interface(
  fn=get_important_facts,
  inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
  outputs="text",
  title="RAG with Llama3",
  description="Ask questions about the proveded context",
)

iface.launch()
Use code with caution
Finally, a user interface is created using Gradio:

get_important_facts is a wrapper function for the rag_chain.
gr.Interface creates the interface with a text box for input and a text output area.
iface.launch() starts the interface, making it accessible in a web browser.
