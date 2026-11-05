import psycopg2

CONN = psycopg2.connect(
        dbname="rag_database",
        user="postgres",
        password="secret",
        host="127.0.0.1",
        port="5433"
)

def print_column_content():
    try:
        cur = CONN.cursor()

        cur.execute("SELECT content FROM document_chunks;")

        rows = cur.fetchall()

        if rows:
            print("--- Content of 'content' column ---")
            for index, row in enumerate(rows, start=1):
                print(f"[{index}]: {row[0]}")
        else:
            print("The table is empty.")

    except Exception as e:
        print(f"error while connecting to the DB: {e}")

    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            CONN.close()

if __name__ == '__main__':
    try:
        print_column_content()
    except Exception as e:
        print(e)