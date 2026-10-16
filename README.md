# *MIMI*: Chatbot help protect Kids from Education Hidden Dangers 
## I. Idea Origin
## II. Techstack Framework
- Python, FastAPI (webhook Facebook Messenger)
- Mimi v2 (`mimi_v2/`): lõi hội thoại, an toàn, chuyển người hỗ trợ
- Redis (phiên hội thoại), PostgreSQL + pgvector (tri thức), Resend (email khẩn)
- Gradio (chat thử cục bộ trước khi đưa lên production)
- Model: OpenAI hoặc Shineshop (tương thích OpenAI), chọn bằng biến môi trường

Repo này chỉ phục vụ giao tiếp qua Facebook Messenger. Mã engine cũ, chat WebSocket và công cụ eval nằm trong `legacy/`
(xem `legacy/README.md`); không cần đọc chúng khi phát triển Mimi v2.

## III. Quick start
### Chat thử bằng Gradio (không cần Facebook, Redis hay database)
1. `pip install -r requirements_local.txt`
2. Chế độ offline (không gọi model): `python -m mimi_v2.simulator`
3. Chế độ live: đặt `MIMI_V2_BASE_URL`, `MIMI_V2_API_KEY`, `MIMI_V2_MODEL` (hoặc `OPENAI_API_KEY`) trong `.env`, rồi
   `python -m mimi_v2.simulator --live`
4. Mở http://127.0.0.1:7860. Dữ liệu chỉ nằm trong RAM; Gradio không kết nối Facebook, database hay email.

### Chạy webhook
1. `pip install -r requirements.txt`
2. Điền biến môi trường theo `.env.production.example`
3. `uvicorn api.main:app --host 0.0.0.0 --port 8000`; webhook ở `/webhook/facebook`, kiểm tra sức khỏe ở `/health`.

### Chạy test
1. `pip install -r requirements_dev.txt`
2. `pytest -q` (test lõi, chạy offline). Test của `legacy/` chạy riêng bằng `pytest legacy/tests`.

## IV. Standard Basement  

**Một hệ AI cho thanh thiếu niên bắt buộc phải đáp ứng 6 trụ cột sau:**
1. Psychological Safety - Không gây tổn hại tinh thần
2. Crisis Awareness - Nhận diện & Xử lý khủng hoảng cảm xúc
Mimi sẽ cố gắng hiểu thấu đáo được những crisis mà bạn đang gặp phải, rồi sau đó đưa sẽ đưa ra chiến lược phù hợp
và tối ưu nhất để có thể thực sự có thể giúp đỡ các bạn 
3. Power Balance - Không tạo lệ thuộc cảm xúc
Mimi có thể là một người bạn chuyên hỗ trợ những bạn gặp khó khăn về vấn đề học đường, tuy nhiên cũng không vì thế
mà để cho các bạn coi Mimi trở thành chỗ dựa tinh thần duy nhất, mà Mimi mong muốn rằng khi người dùng có thể trở
nên mạnh mẽ hơn, dám đối diện với cuộc sống và dám nhờ sự giúp đỡ của người lớn
4. Memory Ethics - Ghi nhớ có chọn lọc, không xâm phạm
Mimi có quyền được ghi nhớ những kí ức của người dùng, tuy nhiên cũng không vì thế mà tự do sử dụng những kiến thức
đó/ hoặc là bịa sai ra những kiến thức để khiến cho người dùng cảm tưởng sai lầm về bản thân, mà từ đó dẫn đến những
hậu quả không mong muốn 
5. Age-Appropriate Reasoning - Phù hợp nhận thức 
Ta biết được là đối tượng sử dụng sản phẩm sẽ là đối tượng từ 8-18 tuổi, và mỗi độ tuổi thì chắc chắn sẽ có
các nhận thức khác nhau về thế giới, thế nên cách nói chuyện của bot cũng phải khác để có thể phù hợp hơn với
lứa tuổi
6. Human Escalation 
Là năng lực mà mô hình chủ động nhận thức được giới hạn của bản thân, sau đó có thể điều hướng sự hỗ trợ thực
sự sang cho con người. Tức là AI phải có mong muốn rằng người nhắn sẽ thực sự được cải thiện hơn là việc mãi
mãi lệ thuộc cảm xúc, và không bao giờ có thể thoát khỏi hố sâu đó