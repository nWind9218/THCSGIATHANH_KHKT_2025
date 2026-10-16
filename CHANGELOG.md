# Changelog

Lịch sử phát triển được dựng từ các commit là tổ tiên của nhánh **design_v2**, tới `3fc628e19f757a578ff572a435dd25f9eb993029`. Khảo sát ngày 14/10/2026. Ngày trong tài liệu dùng ngày tác giả commit và UTC+7; đây không phải ngày xác nhận deploy hoặc phát hành.

Repository tại thời điểm khảo sát **không có Git tag**. `ver01` và `ver(02)` bên dưới là nhãn xuất hiện trong commit message lịch sử, chưa phải tag SemVer. API hiện ghi `1.0.0` trong `api/main.py`; không dùng hằng số này để đặt version cho tất cả mốc sau. Các tiêu đề theo ngày là mốc phát triển, không tự nhận là release đã công bố.

## Tài liệu mới chưa phát hành

- Bộ báo cáo kỹ thuật và giáo án use case Mimi trong `AGENT_DESIGN_DOCS/TEACHING_DESIGN_V2`, dựa trên snapshot design_v2.
- Bổ sung changelog từ Git history, phân biệt mốc code và release.
- Các mục tài liệu mới thuộc thay đổi workspace chưa có commit; không gán chúng vào commit nguồn `3fc628e`.

## 2026 09 22 củng cố Redis và kiểm tra Messenger

### Sửa và bổ sung

- `e4ee7ce`: khóa redis-py 8.0.1, thêm bộ dependency `requirements.lock`, Docker cài lock; thêm socket timeout, keepalive, health check và retry backoff cho Redis. Kiểm tra `/health` được mở rộng sang cả Postgres và Redis; thêm script probe reconnect và webhook. Không kết luận mọi reconnect đã được giải quyết chỉ từ commit message.
- `4c4e945`: Send API dùng header Authorization thay token trong URL; lỗi giao nhận được báo rõ hơn; thêm suppress delivery cho sender probe và test Messenger.
- `3fc628e`: phản hồi webhook xác nhận `probe_delivery_suppressed`, script probe kiểm tra xác nhận này và bổ sung assertion tương ứng.

**Giới hạn còn lại:** webhook vẫn dùng BackgroundTasks trong process, chưa durable queue/dedup; suppress không phải dry run của toàn graph.

## 2026 07 05 thay mặc định model chat

- `5e88638`: đổi default của `OPENAI_CHAT_MODEL` trong `graph/tools.py` từ `gpt-4.1-mini` sang `gpt-5-mini`.
- File env example ở snapshot vẫn đặt `gpt-4.1-mini`; env sẽ ghi đè default. Commit message “booom” được giải thích bằng diff một dòng, không dùng nó để suy ra chức năng mới khác.

## 2026 05 14 tri thức tình huống và escalation

### Bổ sung

- `3170ffe`: thêm CSV tình huống học sinh và `embedd.py` để tạo/nạp student_knowledge; bổ sung search kho này vào memory và node deep; cập nhật prompt tiếng Việt cho safety, intent, info gap, clarification, memory và phản hồi.
- `df8526b`: node deep dò should_do của case RAG để đánh dấu escalation và thêm `tests/test_deep_reasoning_escalation.py`.

### Giới hạn

- Cờ escalation ở node deep chưa đi qua quy trình notify/persist flag của safety handoff. `escalation_reason` chưa có trong state schema. Không gọi mốc này là “handoff đã hoàn thiện”.

## 2026 05 13 đến 14 Messenger và chuyển dịch vụ email

- `7832282`: thêm `api/facebook_webhook.py`, include router Facebook trong app, cập nhật prompt và một số script thử.
- `24653a0`: bổ sung xử lý `#whoami`, `#resume` và allowlist admin cho Messenger; mở rộng log/đường thông báo, dọn script thử và thư mục nạp dữ liệu cũ.
- `dfdcb67`: bổ sung chẩn đoán lỗi SMTP và cấu hình liên quan trong giai đoạn trung gian.
- `608aff7`: chuyển đường email runtime sang Resend HTTP API, bổ sung chẩn đoán, fallback gửi từng recipient và test dịch vụ email. SMTP là lịch sử, không phải đường gửi hiện tại.

## 2026 04 16 thiết kế graph module và API

### Thay đổi kiến trúc

- `ae2022f`: thay kiến trúc agent/prompt cũ bằng các module `graph/`, `memory/`, `prompts/`; `workflow()` trả một graph counseling đang hoạt động. Thêm API FastAPI/WebSocket, bootstrap schema, embedding helpers và tests flow.
- `SYSTEM_PROMPT/` và module agent cũ bị xóa khỏi snapshot tại mốc này. Các prompt runtime hiện nằm trong `prompts/`.

### Đóng gói và cấu hình

- `d4b355b`: Docker, env example, script và cấu hình Render; thêm test integration.
- `715874c`: thay script/cấu hình deploy Render bằng Railway, sửa sử dụng PORT.
- `d258d13`: tiếp tục điều chỉnh Docker và env.
- `2728657`: thêm module CORS/Origin, tích hợp kiểm tra WS và tests cấu hình.

Tên commit liên quan hosting chỉ xác nhận thay đổi mã/cấu hình trong Git, chưa xác nhận dịch vụ hiện đang chạy ở nền tảng nào.

## 2026 04 15 tách bộ system prompt ở kiến trúc cũ

- `fc4e4f0`: thêm SYSTEM_PROMPT có registry/loader và các tài liệu persona/chức năng; sửa agent/tools, workflow và dependency, bỏ workflow_demo.
- Bộ cấu trúc này là một bước trung gian; không dùng nó để giải thích runtime của design_v2 cuối cùng vì ngày 16/04 đã đổi kiến trúc.

## 2026 01 06 đến 07 mốc ver02 và cải tiến bản thử

- `b25d916`: commit mang nhãn `ver(02):finish fixing bugs + temp ver for exam`; thay đổi lớn ở agent/tools, prompt, database, main và workflow.
- `5b03512`: thêm gửi email khẩn cấp ở kiến trúc cũ.
- `3b21002`, `66c2f4d`, `f95ec50`: điều chỉnh tích hợp Ollama/model và Compose trong giai đoạn thử.
- `2fc7253`: chuyển phần trích xuất vấn đề sang tiếng Việt.
- `855416b`, `f806053`, `c4af7e2`: tiếp tục sửa tools/embedding/workflow và main. Các message không mô tả rõ chức năng; không tự đặt tên tính năng không có bằng chứng.

## 2026 01 03 đến 05 mốc ver01 và đóng gói

- `f19d66a`: mở rộng xử lý cảm xúc, rủi ro, quyết định và phản hồi khẩn cấp ở workflow cũ.
- `d52dc59`: thêm logic cache ở kiến trúc cũ; commit này đi vào lịch sử design_v2 qua nhánh được merge, không nằm trên first-parent.
- `28b443f`, `53ab1ed` và các commit cấu hình kế tiếp: thêm/chỉnh Compose, Docker, dependency, port và pool. `8e0d1fd` loại bỏ đường Neo4j ở utils/database.
- `4694659`: commit mang nhãn `ver01: on prod`, sửa state/tools, embedding và workflow. Nhãn “on prod” là nội dung commit, không phải xác nhận deploy độc lập.
- Các merge PR #1 và #2 cùng những commit sửa tiếp theo nằm trong ancestry design_v2. Changelog lấy chúng vì là tổ tiên của nhánh yêu cầu, không lấy head nhánh main làm nguồn hệ thống hiện tại.

## 2025 12 nền tảng ban đầu

- `8876915` ngày 08/12: khởi tạo repository với README rỗng.
- `272a46d`, `725ac6c`, `82cc367` ngày 15–16/12: đưa Python workflow, RAG, embedding, main và Compose vào repo; sửa luồng thử. Commit message đề cập Facebook nhưng không dùng message này để khẳng định đã có webhook API hoàn chỉnh như bản tháng 5.
- `2dbb988` ngày 15/12: thay đổi gitignore liên quan secret.
- `bb77b80`, `9d06533` ngày 26–30/12: bổ sung cấu hình LangGraph/LangSmith, README, client và workflow demo/monitoring ở kiến trúc cũ.
- `4d48ede` ngày 30/12: dọn cache/bytecode/checkpoint artifacts khỏi Git và cập nhật ignore.

## Bằng chứng và quy ước cho các bản sau

Snapshot gồm **56 commit reachable**, trong đó **53 commit first-parent**. Phụ lục `AGENT_DESIGN_DOCS/TEACHING_DESIGN_V2/git-history.json` và `GIT_HISTORY.md` lưu đầy đủ danh sách, ngày, parent và file thay đổi để tra các commit có message mơ hồ.

Khi phát hành thật, nhóm dự án nên quyết định quy ước version, tạo tag trên commit đã được kiểm tra và ghi Added/Changed/Fixed cùng breaking changes, migration và validation. Đề xuất này chưa tạo tag hay release. Không tự gán số SemVer giả cho các mốc lịch sử.

Lệnh tái kiểm tra: `git log design_v2 --date=iso-strict --format='%H %aI %s'`, `git log design_v2 --first-parent`, `git show <commit> -- <file>`, `git tag --list`. Nhánh có thể tiến lên sau ngày khảo sát; bộ tài liệu này vẫn ghim snapshot `3fc628e`.
