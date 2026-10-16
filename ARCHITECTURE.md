# Kiến trúc

Repo phục vụ Mimi trên **Facebook Messenger**, cộng **Gradio** để chat thử trước khi lên production.

## Cấu trúc

```
api/
  main.py               FastAPI: /health, /webhook/facebook
  facebook_webhook.py   Xác minh chữ ký, tách event, gọi engine, gửi lại qua Graph API; lệnh admin #accept, #resume, #whoami
mimi_v2/
  core.py               CounselingEngine: safety, assessment, reply, chuyển người hỗ trợ
  contracts.py          Schema và chuẩn hóa đầu ra model
  prompts.py            Prompt của v2
  models.py             Adapter OpenAI-compatible (Responses / Chat Completions), chọn endpoint bằng MIMI_V2_BASE_URL
  store.py              InMemoryStore (Gradio/test)
  production.py         ProductionStore: Redis (phiên), pgvector (tri thức), email khẩn
  simulator.py          Gradio chat thử (offline hoặc live)
memory/                 log_emergency, send_emergency_email
utils/                  Kết nối Postgres/Redis, embeddings
scripts/                Khởi tạo schema, probe webhook/Redis production
embedd.py               Nạp dữ liệu ca học sinh vào pgvector
legacy/                 Engine cũ, WebSocket, bộ eval (xem legacy/README.md)
tests/                  Test lõi, chạy offline
```

## Luồng một tin nhắn

1. Facebook gọi `POST /webhook/facebook`; chữ ký `X-Hub-Signature-256` được kiểm tra bằng `FACEBOOK_APP_SECRET`.
2. Mỗi event có text/postback được đưa vào `BackgroundTasks`; webhook trả 200 ngay.
3. `handle_facebook_event` xử lý lệnh admin, nếu không thì gọi `get_production_engine().respond(psid, text)`.
4. Engine lấy khóa hội thoại trong Redis, nạp phiên, chạy safety, assessment, reply, rồi lưu phiên.
5. Nếu cần người hỗ trợ: ghi log, gửi email (Resend), phiên ở trạng thái `pending`. Admin gõ `#accept <PSID>` để tiếp nhận
   (bot im lặng) và `#resume <PSID>` để bot trả lời lại.
6. Câu trả lời được cắt theo giới hạn 2.000 ký tự và gửi qua Graph API.

## Chọn model

- Mặc định: OpenAI (`https://api.openai.com/v1`, key `OPENAI_API_KEY`).
- Shineshop: đặt `MIMI_V2_BASE_URL=https://api.shineshop.dev/v1`, `MIMI_V2_API_KEY`, `MIMI_V2_MODEL`.
  Host ngoài danh sách cho phép bị từ chối, và key OpenAI không bao giờ gửi sang endpoint tùy chỉnh.
- Kiểm tra an toàn/assessment cần structured output; model mới phải được xác minh rồi thêm vào `_supports_structured_output`.
- Embeddings vẫn dùng `OPENAI_API_KEY` / `OPENAI_API_EMBEDDING_KEY`.

## Gradio

`python -m mimi_v2.simulator [--live]` dùng cùng `CounselingEngine` với production nhưng thay storage bằng `InMemoryStore`
và không import connector production. Có thể mô phỏng lỗi model, storage, thông báo.

## Triển khai

`Dockerfile` cài `requirements.lock` và chạy `uvicorn api.main:app`. `.dockerignore` loại `legacy/`, `tests/` và tài liệu
khỏi image. Biến môi trường mẫu: `.env.production.example`.
