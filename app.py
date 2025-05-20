from rag_engine import retrieve_similar_docs, generate_answer_ollama

def main():
    print("Loading RAG-based QA system...")
    while True:
        query = input("\nAsk a technical question (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        relevant_docs = retrieve_similar_docs(query)
        context = "\n\n".join(relevant_docs)
        answer = generate_answer_ollama(context, query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
