import psycopg2
from pgvector.psycopg2 import register_vector
from llama_index.embeddings.ollama import OllamaEmbedding


def semantic_search(query_text: str, top_k: int = 3):
    print(f"1. Векторизиране на въпроса: '{query_text}'...")

    # Инициализиране на същия модел, с който са записани чънковете
    embed_model = OllamaEmbedding(
        model_name="bge-m3",
        base_url="http://localhost:11434"
    )

    # Генериране на вектор за въпроса
    query_vector = embed_model.get_text_embedding(query_text)

    print("\n2. Търсене в PostgreSQL (pgvector)...")
    try:
        # Свързване към базата данни с твоите настройки
        conn = psycopg2.connect(
            dbname="rag_database",
            user="postgres",
            password="secret",
            host="127.0.0.1",
            port="5433"
        )

        # Регистриране на векторния тип
        register_vector(conn)
        cur = conn.cursor()

        # SQL заявка за семантично търсене чрез косинусово разстояние (<=>)
        # 1 - (embedding <=> query_vector) ни дава similarity score, където стойност по-близка до 1 е по-добро съвпадение
        search_query = """
                       SELECT content, 1 - (embedding <=> %s::vector) AS similarity_score
                       FROM document_chunks
                       ORDER BY embedding <=> %s::vector
                           LIMIT %s; \
                       """

        # Подаваме вектора два пъти (един път за SELECT изчислението и един път за ORDER BY сортирането)
        cur.execute(search_query, (query_vector, query_vector, top_k))
        results = cur.fetchall()

        print(f"\n--- Топ {top_k} най-релевантни чънка ---")
        for index, (content, score) in enumerate(results, start=1):
            print(f"\n[Резултат {index}] Similarity Score: {score:.4f}")
            print(f"Съдържание: {content.strip()}")

    except Exception as e:
        print(f"Грешка при изпълнение на търсенето: {e}")
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()


if __name__ == '__main__':
    # Тестов въпрос към текстовете за Димчо Дебелянов
    sample_question = "Кои са образите в \"Аз искам да те помня все така\"?"
    semantic_search(sample_question, top_k=3)