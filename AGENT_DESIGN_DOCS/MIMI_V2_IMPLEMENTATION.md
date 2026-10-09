# Mimi v2 — đặc tả triển khai và nghiệm thu

Ngày: 09/10/2026. Nhánh triển khai: `codex/mimi-v2`, nền `design_v2` (`3fc628e`).
Phạm vi đã được người dùng duyệt: lõi hội thoại, API/model, Gradio giả lập Messenger và conversation eval. Người dùng đã cho phép dùng lại khóa OpenAI hiện có; probe Responses gpt-5-mini trả HTTP 200. Không triển khai production hoặc merge khi chưa có bằng chứng nghiệm thu.

## Thiết kế

- Giữ workflow legacy làm baseline mặc định. V2 được chọn bằng `MIMI_ENGINE=v2`; không đổi webhook URL, chữ ký, Page credential hoặc Facebook Send API.
- Package `mimi_v2` không import database, Facebook hoặc email khi dùng local. Lõi `CounselingEngine(model, store).respond(user_id, text)` dùng cùng hợp đồng ở Messenger, Gradio và eval.
- Model adapter: `ModelConfig`, `ModelReply(text, usage, metadata)`, `OpenAIModel.complete(messages, purpose)`. Chọn Chat Completions/Responses, model, reasoning, verbosity, timeout qua cấu hình. `store=False`; lịch sử thuộc application, không nối previous_response_id giữa người dùng.
- Store có `load`, `save`, `search`, `notify`. Local dùng bộ nhớ tách biệt; production adapter đọc history/topic/takeover Redis cũ và giữ metadata v2 trong key riêng. Không reset dữ liệu.
- Canonical history được nạp một lần, thêm user một lần, thêm assistant một lần và ghi snapshot 20 tin nhắn. Không reducer cộng snapshot. Khóa theo conversation để tránh lost update.
- Đánh giá an toàn trước RAG; assessment được kiểm tra kiểu dữ liệu và giữ known/context/need/missing_fields/question_focus/core_issue/role. Mỗi bước sinh câu trả lời nhận lịch sử và assessment.
- Persona chung: trưởng thành, gần gũi, rõ ràng, không dạy bảo; tôn trọng listen-only và phản hồi đóng. Parent role chỉ từ khai báo/ngữ cảnh rõ, không cấp quyền xem dữ liệu.
- Handoff none/pending/accepted: cảnh báo không đồng nghĩa tiếp nhận. Pending vẫn có hỗ trợ an toàn; accepted dừng bot. Gửi thông báo thất bại không làm mất phản hồi. Thao tác accept/resume Messenger chỉ từ PSID admin cấu hình, không từ role do model suy ra.
- Gradio chỉ loopback, không share. Fake model mặc định; live OpenAI opt-in. Không kết nối Redis/Postgres production, không gửi Facebook/email/alert thật.
- Eval chạy từng conversation với store riêng, ghi dataset/prompt/config hashes, output/usage/latency/error, tiêu chí tự động và human review. Baseline phải dùng thực sự workflow legacy trong sandbox, không đổi tên engine fake thành baseline. Dataset người dùng chưa có; fixture smoke không chứng minh chất lượng tư vấn.

## BLOCK và Definition of Done

1. History không trùng/đảo; dữ liệu cũ đọc được; context không rò giữa người dùng.
2. Assessment không bị mất field; JSON/safety không hợp lệ có fallback rõ.
3. Hiểu câu tiếp nối, correction, closing, listen-only; parent role không cấp quyền.
4. Nguy cơ ở safety/assessment đi cùng handoff; lỗi model/DB/notify không làm mất hỗ trợ.
5. Adapter hai API có test normalization, capability validation, timeout, refusal/incomplete và redaction.
6. Gradio/eval có test chứng minh không gọi production services. UI có kiểm tra pass/fail.
7. Eval chạy dataset nhiều lượt, lỗi/skipped không được tính pass, có report và đánh giá người duyệt.
8. Hợp đồng webhook/Send API legacy vẫn được kiểm tra hồi quy; feature flag mặc định legacy, rollback không reset memory.
9. Có lệnh local/eval xác minh được và biên nhận test, live smoke API và UI.

## OUT OF SCOPE

WebSocket, polish UI, tự đổi model mặc định sau eval chưa duyệt, tự rollout/merge/deploy; multi-agent sản phẩm, voice và OCEAN thuộc PR follow-up. Không thay schema hoặc dữ liệu production trong quá trình test.

## Phân công và thứ tự

Director: spec, production store/feature flag, regression/hardening, evidence và docs.
Subagent core: core/prompts/store và unit tests.
Subagent adapter: models và unit tests.
Subagent local/eval: Gradio/dataset/jobs/reports và isolation tests.
Tích hợp sau contract thống nhất; test offline trước, live OpenAI chỉ dữ liệu synthetic; không chạy production probe.

## Verification

Repo hướng dẫn `pytest -q`. Chạy test unit đã cô lập trước; suite E2E/deep integration cũ có phụ thuộc services thật nên không được chạy mù. Venv local `.venv/Scripts/python.exe`; Docker Python 3.11. Optional Gradio dependencies nằm riêng `requirements_local.txt`, không thay lock production vì công cụ local.
## Biên nhận thực thi 09/10/2026

- Nhánh hiện tại `codex/mimi-v2`; không merge, deploy hoặc bật Facebook v2.
- Khóa OpenAI hiện có dùng lại theo yêu cầu. GET models và probe Responses HTTP 200; adapter Responses và Chat Completions đều trả nội dung hợp lệ với `gpt-5-mini`, minimal reasoning, `store=False`. Chỉ gửi dữ liệu synthetic.
- Lệnh `python -m pytest -q tests/test_v2_models.py tests/test_v2_core.py tests/test_v2_production.py tests/test_v2_local_eval.py tests/test_v2_messenger_routing.py tests/test_facebook_webhook_delivery.py tests/test_resend_email_service.py`: **94 passed** ở thời điểm ghi biên nhận. Không chạy suite integration cũ cần dịch vụ thật. Số này không bao gồm test bổ sung sau biên nhận.
- Live core hai lượt với InMemoryStore: giữ role student và thông tin lớp 8, assessment hợp lệ, không diagnostic lỗi; latency 12.577s và 8.195s. Vẫn đặt câu hỏi ở lượt explicit listen-only: cần tinh chỉnh prompt và kiểm chứng lại, không tính là hành vi đạt nghiệm thu.
- Production adapter có timeout I/O, RAG và notification; model timeout tối đa 60s. Không lấy được distributed lock thì không ghi state hoặc trả lời giả như đã xử lý bình thường; Messenger ghi nhận lỗi. Cần quan sát trường hợp lock contention trước rollout.
- Dataset nghiệm thu của người dùng và human review chưa có. Fixture/offline smoke không chứng minh v2 thông minh hơn baseline hoặc nhanh hơn; chưa có kết luận về ưu thế tốc độ của Responses.
- Sau tinh chỉnh LISTEN_MODE: lượt live listen-only không đặt câu hỏi, giữ known, assessment hợp lệ và không diagnostic lỗi (7.033s, 1.756 tokens). Đây chỉ là một mẫu smoke, không bảo đảm mọi tình huống đều đạt.
- Suite trên chạy lại sau hardening: **95 passed**. Gradio đã kiểm tra rendered conversation, pending → accepted, resume và fault model; comparison smoke 10 lượt không lỗi, tất cả pending human review. Hướng dẫn riêng tại `MIMI_V2_LOCAL_TESTING.md`.
- GitHub CLI chưa đăng nhập: chưa tạo/push PR qua CLI, chưa merge. Các thay đổi đang nằm trên nhánh local `codex/mimi-v2`.

## Cập nhật dataset theo yêu cầu người dùng

Người dùng cho phép tác giả tự đề xuất 1.000 case thay việc chờ transcript người
dùng. Đã materialize `eval_datasets/v2_acceptance_1000.json`: 20 nhóm, 200 tình
huống gốc × 5 trở ngại hành vi, 4.400 lượt user; chia 600/200/200 theo tình huống
gốc, có lời kể dài ở cả ba tập. Dữ liệu synthetic và rubric không có chứng nhận
lâm sàng. Xem `eval_datasets/README.md` và coverage JSON để biết giới hạn.

Runner chỉ hỗ trợ setup/actions/faults trên InMemoryStore; accept/resume/reset
và lỗi model/đọc phiên/cảnh báo có trace thực, không lấy lời bot làm bằng chứng.
Suite unit/hồi quy mở rộng: **108 passed**. Full offline comparison kiểm tra wiring,
không phải nghiệm thu trí thông minh. Không còn blocker thiếu file dataset;
nghiệm thu hành vi còn cần chạy model thật và review theo tiêu chí đã đề xuất.

## Trạng thái hiện tại — full live eval đã được duyệt

Các biên nhận phía trên là lịch sử, không phải trạng thái GitHub hiện tại.
GitHub đã đăng nhập, nhánh `codex/mimi-v2` đã push và draft PR #3 đã tạo:
https://github.com/nWind9218/THCSGIATHANH_KHKT_2025/pull/3.
Không merge, deploy hoặc bật Facebook v2.

- Pilot cân bằng 20 nhóm đã hoàn tất: 583 request, USD ước tính từ usage
  `0.20937555`. Chưa đối soát invoice hoặc số dư tài khoản.
- Safety/assessment đã dùng strict Structured Outputs, known wire-array được
  chuẩn hóa về dictionary trong state. Preflight sáu assessment từng lỗi đều
  hợp lệ; chi phí usage ước tính `0.00402225`. Không coi đây là nghiệm thu tư vấn.
- Suite unit/hồi quy đạt 159 test; batch suite cuối đạt 10 test, bao gồm quota
  đồng thời, checkpoint/resume, factory failure và không replay lượt đã xong.
- Người dùng cho phép chạy toàn bộ 1.000 case baseline/v2, tổng 8.800 lượt.
  Job `191d033deed0430d83d45cbceaaa07ca` khởi chạy lúc
  `2026-10-09T05:20:58Z`, concurrency 8, source tại commit `17e8f95`.
  Kết quả local được lưu dần tại `eval_results/full_1000`; manifest khóa hash
  dataset/config/source. Xem `MIMI_V2_BATCH_EVAL.md` để resume.
- Lỗi từng case không dừng các case độc lập. Quota/credential/persistence mở
  circuit bảo vệ, giữ kết quả và lượt chưa chạy để resume; không coi lỗi hoặc
  unprocessed là pass. Không tự mua credits hoặc replay request không chắc đã
  được tính phí. Mã runtime không thay đổi trong khi job đang chạy.
- Full live report và qualitative review còn đang chờ; chưa có bằng chứng đủ
  để kết luận v2 thông minh hơn, nhanh hơn hoặc đạt tiêu chuẩn triển khai.
