# legacy/

Mã không còn nằm trong luồng Facebook + Gradio. Giữ lại để tham khảo và để chạy lại bộ eval; không bảo đảm chạy được.

| Thư mục/file | Nội dung |
|---|---|
| `graph/`, `prompts/`, `workflow.py`, `main.py`, `langgraph.json` | Engine LangGraph cũ và chat CLI |
| `api/chat_ws.py`, `api/admin_ws.py`, `api/cors_config.py` | Chat WebSocket và giám sát giáo viên (đã thay bằng email + `#accept` qua Facebook) |
| `memory_redis.py`, `memory_extras.py`, `memory_compat.py` | Phần `memory/` chỉ engine cũ dùng |
| `eval_tools/` | `batch_eval`, `pilot`, `evaluation`, `datasets`, `baseline`, `assessment_preflight` |
| `eval_datasets/` | Bộ 1.000 hội thoại synthetic và catalog |
| `scripts/` | Script phục vụ engine cũ và eval |
| `design/` | Notebook và dataset ca học sinh (dùng cho `embedd.py`) |
| `tests/` | Test của các phần trên; chạy `pytest legacy/tests` |

Chạy công cụ eval từ thư mục gốc repo, ví dụ `python -m legacy.eval_tools.batch_eval --help`. Lưu ý: checkpoint eval cũ
ràng buộc hash mã nguồn đã đóng băng, nên không resume được job cũ bằng bản mã này.
