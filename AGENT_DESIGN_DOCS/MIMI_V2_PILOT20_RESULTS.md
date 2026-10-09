# Live pilot 20 nhóm — 09/10/2026

Được người dùng cho phép chạy trước khi quyết định mở rộng. Đúng 20 case, mỗi
nhóm một case development; mỗi case chạy baseline và v2 với cùng `gpt-5-mini`,
Responses API, reasoning minimal, verbosity low, cap 2048, timeout 60s.
Không dùng holdout, hội thoại thật hoặc production services; không tự retry.

## Chi phí và phạm vi

| Engine | Lượt gọi API | Input tokens | Cached input | Output tokens | USD theo usage |
|---|---:|---:|---:|---:|---:|
| Baseline | 374 | 261.676 | 0 | 26.470 | 0,11835900 |
| V2 | 209 | 162.167 | 10.112 | 26.375 | 0,09101655 |
| Tổng | 583 | 423.843 | 10.112 | 52.845 | **0,20937555** |

Giá standard kiểm tra từ [trang model OpenAI](https://developers.openai.com/api/docs/models/gpt-5-mini):
input 0,25 USD, cached input 0,025 USD, output 2 USD / triệu token.
Cached tokens không tính hai lần; reasoning đã thuộc output tokens.
Không có lượt thiếu usage, lỗi API, lỗi observer/ghi ledger. Thời gian 487,85s.

Đây là chi phí tính từ usage quan sát, **không phải số dư/hóa đơn đã đối soát**.
OpenAI Platform Usage trong trình duyệt yêu cầu đăng nhập; không có quyền đọc số
tiền tài khoản thực trừ. Không có kết luận về giảm giá, thuế hay billing riêng.
Chi phí trên chỉ cho đợt pilot này, không cộng các smoke probe đã chạy trước đó.

## Kết quả không được coi là nghiệm thu

40 case/config runs, 176 lượt user. Baseline: 86 pending review, 2 failed checks.
V2: 82 pending review, 6 lượt `assessment_unavailable_or_invalid`. Không có pass
ngữ nghĩa do chưa nộp human review; không tự biến pending thành pass.

Hai failed checks baseline: `vn-v2-0113`, `vn-v2-0858`, lượt index 2, support mode.
Sáu lỗi v2 (index bắt đầu 0):

- `vn-v2-0215`: 2.
- `vn-v2-0409`: 0.
- `vn-v2-0536`: 1 và 2.
- `vn-v2-0557`: 1 và 4; lượt 4 cũng trượt max_questions.

Rà soát sơ bộ còn thấy handoff hơi rộng và lặp template qua các lượt; một số
correction/listen-only tốt hơn baseline. Đây là quan sát để tìm lỗi, không phải
chứng nhận chuyên môn. Cần làm rõ lỗi assessment, cải thiện phản hồi theo ngữ
cảnh và review trước khi rollout. Giá v2 thấp hơn trong mẫu không đủ chứng minh
v2 tốt hơn; một số fallback cũng dùng ít model calls hơn phản hồi bình thường.
Đợt này không so Responses với Chat Completions nên không kết luận API nhanh hơn.

## Biên nhận

- Report local: `eval_results/live_pilot_20/6de33600246a4016b6e0d8278d72fdc5.json`.
- Ledger `usage.jsonl`, `cost_summary.json`, pilot dataset và checkpoint cùng thư mục.
- Artifacts hội thoại không push lên GitHub; repo chỉ lưu bản tổng kết này.
- 119 unit/regression tests đạt, gồm test metering cho output incomplete/refusal.
- [PR draft #3](https://github.com/nWind9218/THCSGIATHANH_KHKT_2025/pull/3), base `design_v2`.
- Chưa chạy paid rerun/full 1.000 case; chưa merge hoặc thay Facebook đang chạy.
