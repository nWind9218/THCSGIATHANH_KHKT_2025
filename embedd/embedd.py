import re
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from langchain_ollama import OllamaEmbeddings

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER = os.getenv("DB_USERNAME")
PG_PASS = os.getenv("DB_PASSWORD")

try:
    connection = psycopg2.connect(
        host=PG_HOST_AI,
        port=PG_PORT_AI,
        user=PG_USER,
         password=PG_PASS,
        database="mydb"
    )
    cursor = connection.cursor()
    
    # Đọc CSV file
    print(f"\n{'='*50}")
    print("📂 Đang đọc file bot_knowledge.csv...")
    print(f"{'='*50}")
    
    df = pd.read_csv('bot_knowledge.csv')
    
    # Hiển thị thông tin
    print(f"✅ Đã đọc {len(df)} dòng từ CSV")
    print(f"📋 Columns: {list(df.columns)}")
    
    # Loại bỏ các dòng có problem rỗng
    df = df.dropna(subset=['problem'])
    print(f"📊 Sau khi loại bỏ problem rỗng: {len(df)} dòng")
    
    # Khởi tạo embedding model
    print(f"\n🤖 Khởi tạo Ollama Embedding Model...")
    embedding_model = OllamaEmbeddings(model="bge-m3:latest", base_url=OLLAMA_HOST
)
    
    # Lưu cặp (index, vector) để mapping đúng
    successful_embeddings = []
    
    print(f"\n{'='*50}")
    print("🔄 Bắt đầu embedding...")
    print(f"{'='*50}\n")
    
    for idx, row in df.iterrows():
        problem = row['problem']
        
        if isinstance(problem, str) and problem.strip():
            try: 
                vector = embedding_model.embed_query(problem)
                successful_embeddings.append((idx, vector))
                print(f"✅ Embedded {idx+1}/{len(df)}: {problem[:50]}...")
            except Exception as e:
                print(f"❌ Lỗi embedding dòng {idx+1}: {e}")
        else:
            print(f"⚠️ Bỏ qua dòng {idx+1}: problem không hợp lệ")
    
    # Insert vào database
    print(f"\n{'='*50}")
    print("💾 Đang lưu vào database...")
    print(f"{'='*50}\n")
    
    inserted_count = 0
    for idx, vector in successful_embeddings:
        try:
            row = df.iloc[idx]
            
            # Parse must_not_do nếu là string
            must_not_do = row.get('must_not_do', '')
            if isinstance(must_not_do, str):
                # Tách chuỗi thành array nếu có dấu phân cách
                must_not_do = must_not_do if must_not_do else ''
            
            # Parse language_signals nếu là string
            language_signals = row.get('language_signals', '')
            if isinstance(language_signals, str):
                language_signals = language_signals if language_signals else ''
            
            cursor.execute(
                """INSERT INTO bot_knowledge(
                    problem, solution, tone, must_not_do, level, 
                    language_signals, embedding, self_harm, violence, urgency
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    row['problem'],
                    row.get('solution', ''),
                    row.get('tone', ''),
                    must_not_do,
                    int(row.get('level', 0)),
                    language_signals,
                    vector,
                    bool(row.get('self_harm', False)),
                    bool(row.get('violence', False)),
                    row.get('urgency', 'normal')
                )
            )
            inserted_count += 1
            print(f"✅ Đã lưu dòng {idx+1}: {row['problem'][:50]}...")
        except Exception as e:
            print(f"❌ Lỗi insert dòng {idx+1}: {e}")
            print(f"   Data: {row.to_dict()}")
    
    connection.commit()
    print(f"\n{'='*50}")
    print(f"🎉 Hoàn thành! Đã lưu {inserted_count}/{len(successful_embeddings)} dòng vào database")
    print(f"{'='*50}")
    
    print(f"\n{'='*50}")
    print("🎉 Hoàn thành tất cả!")
    print(f"{'='*50}")

except psycopg2.Error as e:
    print("❌ Lỗi kết nối PostgreSQL:")
    print(e)
except Exception as e:
    print("❌ Lỗi khác:")
    print(e)
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'connection' in locals():
        connection.close()
    print("\n✅ Đã đóng kết nối database")