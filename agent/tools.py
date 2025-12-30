import json
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import OllamaEmbeddings

embedd = OllamaEmbeddings(model="bge-m3:latest", base_url="http://localhost:11434")

PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER= os.getenv("DB_USERNAME")
PG_PASS= os.getenv("DB_PASSWORD")

async def rag(question: str, bot_type: str, top_k: int = 3) -> str:
    try:
        question_vector = embedd.embed_query(question)
        connection = psycopg2.connect(
            host=PG_HOST_AI,
            port=PG_PORT_AI,
            user=PG_USER,
            password=PG_PASS,
            database="bot_knowledge"
        )
        cursor = connection.cursor()
        query = """
        Select question, answer, bot_type,
        1 - (embedding <=> %s::vector) as similarity
        FROM bot_knowledge
        WHERE bot_type = %s::vector
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        cursor.execute(query, (question_vector, bot_type, question_vector, top_k))
        results = cursor.fetchall()

        if not results: 
            return json.dumps({ 
                "message":""
            }, ensure_ascii=False)
        formatted_results = []
        for row in results:
            formatted_results.append({
                "question": row[0],
                "answer": row[1],
                "bot_type":row[2],
                "similarity": float(row[3])
            })
        return json.dumps({
            "status":"success",
            "results": formatted_results,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status":"error",
            "message":str(e)
        }, ensure_ascii=False)
    finally:
        connection.close()
        cursor.close()

@tool("human_assistance")
async def hitl() -> str:
    pass