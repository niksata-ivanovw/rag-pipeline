import psycopg2
from pgvector.psycopg2 import register_vector
from llama_index.embeddings.ollama import OllamaEmbedding


def semantic_search(query_text: str, top_k: int = 3):
    print(f"Vectorizing the question: '{query_text}'...")

    embed_model = OllamaEmbedding(
        model_name="bge-m3",
        base_url="http://localhost:11434"
    )

    # generate the vector
    query_vector = embed_model.get_text_embedding(query_text)

    print("Searching in PostgreSQL (pgvector)...")
    try:
        # connecting to the DB
        conn = psycopg2.connect(
            dbname="rag_database",
            user="postgres",
            password="secret",
            host="127.0.0.1",
            port="5433"
        )

        # register the vector
        register_vector(conn)
        cur = conn.cursor()

        # SQL query for semantic search using cos distance(<=>)
        search_query = """
                       SELECT content, 1 - (embedding <=> %s::vector) AS similarity_score
                       FROM document_chunks
                       ORDER BY embedding <=> %s::vector
                           LIMIT %s; \
                       """

        # pass the vector 2 times (1 - select query, 2 - order by sorting)
        cur.execute(search_query, (query_vector, query_vector, top_k))
        results = cur.fetchall()

        print(f"--- top {top_k} most relevant chunks ---")
        for index, (content, score) in enumerate(results, start=1):
            print(f"[result {index}] Similarity Score: {score:.4f}")
            print(f"content: {content.strip()}")

    except Exception as e:
        print(f"error while searching: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    # sample_question = "Кои са образите в \"Аз искам да те помня все така\"?"
    sample_question = "Колко точки получава човекът на позиция 4?"
    semantic_search(sample_question, top_k=3)