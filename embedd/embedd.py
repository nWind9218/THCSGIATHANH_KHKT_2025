import re
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
from langchain_community.document_loaders import UnstructuredExcelLoader


load_dotenv()

PG_HOST_AI = "localhost"
PG_PORT_AI = 5432
PG_USER= os.getenv("DB_USERNAME")
PG_PASS= os.getenv("DB_PASSWORD")

try:
    connection = psycopg2.connect(
        host=PG_HOST_AI,
        port=PG_PORT_AI,
        user=PG_USER,
        password=PG_PASS,
        database= "mydb"
    )
    cursor = connection.cursor()
    
    excel_01 = pd.read_excel('data_embedd.xlsx', sheet_name="Bullying")
    excel_02 = pd.read_excel('data_embedd.xlsx', sheet_name="Pressure")
    excel_03 = pd.read_excel('data_embedd.xlsx', sheet_name="Conflicts")
    excel_04 = pd.read_excel('data_embedd.xlsx', sheet_name="Online_Safety")
    excel_05 = pd.read_excel('data_embedd.xlsx', sheet_name="Sadness_Loneliness")
    
    from langchain_ollama import OllamaEmbeddings
    embedding = OllamaEmbeddings(model="bge-m3:latest", base_url="http://localhost:11434")
    
    excel_lst = [excel_01, excel_02, excel_03, excel_04, excel_05]
    bot_type_lst = ["Bắt nạt học đường","Áp lực","Mâu thuẫn","An toàn không gian mạng","Nỗi buồn cô đơn"]
    
    for i, excel in enumerate(excel_lst):
        excel = excel.dropna(subset=['question', 'answer'])
        question = list(excel["question"])
        answer = list(excel["answer"])
        bot = bot_type_lst[i]
        
        print(f"\n{'='*50}")
        print(f"Đang xử lý: {bot}")
        print(f"Số câu hỏi: {len(question)}")
        print(f"{'='*50}")
        
        # Lưu cặp (index, vector) để mapping đúng với question/answer
        successful_embeddings = []
        
        for j, q in enumerate(question):
            if isinstance(q, str) and q.strip():
                try: 
                    vector = embedding.embed_query(q)
                    successful_embeddings.append((j, vector))
                    print(f"✅ Embedded câu {j+1}/{len(question)}")
                except Exception as e:
                    print(f"❌ Lỗi embedding câu {j+1}: {e}")
            else:
                print(f"⚠️ Bỏ qua câu {j+1}: giá trị không hợp lệ")

        # # Insert vào database
        # print(f"\nĐang lưu vào database...")
        # inserted_count = 0
        # for j, vector in successful_embeddings:
        #     try:
        #         quest = question[j]
        #         ans = answer[j]
        #         cursor.execute(
        #             "INSERT INTO bot_knowledge(question, answer, embedding, bot_type) VALUES (%s,%s,%s,%s)",
        #             (quest, ans, vector, bot)
        #         )
        #         inserted_count += 1
        #     except Exception as e:
        #         print(f"❌ Lỗi insert câu {j+1}: {e}")
        
        # connection.commit()
        # print(f"✅ Đã lưu {inserted_count}/{len(successful_embeddings)} câu vào database")
    
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