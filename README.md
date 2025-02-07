# RAG with Llama3 and Gradio

This project demonstrates a Retrieval Augmented Generation (RAG) system using the `Ollama` language model, `Langchain`, and the `Gradio` library for a user interface. It allows users to ask questions about a specific Wikipedia page (Ohiya) and get answers based on the information extracted from that page.

## How it Works

1. **Data Loading and Preparation:**
   - The content of the Wikipedia page about Ohiya is loaded using `WebBaseLoader` from `langchain_community`.
   - The loaded content is split into smaller chunks using `RecursiveCharacterTextSplitter` from `langchain`.
   - Embeddings are generated for each chunk using `OllamaEmbeddings` from `langchain_community`, and these embeddings are stored in a `Chroma` vector database for efficient searching.

2. **Ollama LLM Interaction:**
   - A function `ollama_llm` is defined to handle interactions with the Ollama Llama3 model. 
   - It takes a question and context as input, formats them into a prompt, and sends the prompt to the model.
   - The model's response is then extracted and returned.

3. **RAG Pipeline:**
   - A retriever is created from the vector database to search for relevant documents based on user queries.
   - The `rag_chain` function orchestrates the RAG process:
      - It takes a question as input.
      - Retrieves relevant documents using the retriever.
      - Formats the context from the retrieved documents.
      - Calls the `ollama_llm` function with the question and context to get the answer.

4. **Gradio Interface:**
   - A user-friendly interface is built using `Gradio`.
   - The `get_important_facts` function acts as a wrapper for the `rag_chain` function.
   - `gr.Interface` creates the interface with a text box for user input and a text output area for displaying the answers.
   - `iface.launch()` starts the interface, making it accessible in a web browser.

## Dependencies

- `gradio`
- `ollama`
- `beautifulsoup4` (although imported, it's unused in this code)
- `langchain`
- `langchain_community`
- `chromadb`

## Usage

1. Install the required dependencies:
