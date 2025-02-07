from retriever import retrieve_top_document
from generator import generate_answer

def main():
    print("Welcome to Simple RAG! Type 'exit' to quit.")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() == "exit":
            break
        
        # Retrieve relevant document
        retrieved_docs = retrieve_top_document(query)
        context = retrieved_docs[0]["text"] if retrieved_docs else "No relevant document found."
        
        # Generate answer using context
        answer = generate_answer(query, context)
        print("\nGenerated Answer:", answer)

if __name__ == "__main__":
    main()

