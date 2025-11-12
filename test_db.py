# db_connect.py
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        port=os.getenv("DB_PORT")
    )

if __name__ == "__main__":
    try:
        conn = get_db_connection()
        print("✅ Connected to PostgreSQL successfully!")
        conn.close()
    except Exception as e:
        print("❌ Database connection failed:", e)
