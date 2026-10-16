# Danh mục rubric Mimi v2 — 2026-10-17

Phạm vi: gói eval offline trong PR #4. Duyệt rubric không phải kết quả eval chatbot production. Hai bộ có anchor đã duyệt hiện là Q2/Q3 và persona P1–P6. Q4 có cấu trúc được duyệt và anchor/contract draft riêng để review, chưa áp dụng vào validator.

| Nhóm | Đánh giá | Trạng thái |
|---|---|---|
| Q2 — chọn hỗ trợ | Nhu cầu giao tiếp hiện tại, reflect-first, advice-when-appropriate; listen-only và no-questions độc lập; phân biệt than và xin giải pháp | Đã duyệt; anchor 0–3, validator và replay offline. Chưa đo semantic accuracy bằng human gold. |
| Q3 — chất lượng giao tiếp | Tự nhiên, tôn trọng, hợp lứa tuổi/bối cảnh, không gây áp lực hoặc lặp khuôn; ngắn vẫn có thể đạt 3 | Đã duyệt; anchor 0–3, validator và replay offline. Không đủ để kết luận toàn bộ persona. |
| Persona P1–P6 | Ấm áp có chú ý cụ thể; tiếng Việt tự nhiên; sắc thái linh hoạt; nhịp đối thoại; nhất quán/thích nghi; trung thực/ranh giới | Chủ sản phẩm duyệt rubric ngày 2026-10-17. Chưa có persona observer output/scorer/gate. 17 tình huống/51 phản hồi minh họa vẫn UNREVIEWED. |
| Q4 — Useful & Safe Guidance | Bám tình huống/nguy cơ; không tăng rủi ro; khả thi; thứ tự can thiệp; kết nối hỗ trợ; quyền chủ động | Sáu chiều, applicability ba trạng thái và MIN được duyệt ở mức thiết kế. Anchor chi tiết và contract draft đang review; validator/gold chưa đổi. Ưu tiên cặp 03–04. |
| Q1/Q5/Q6 và Q4 executable acceptance | Dữ kiện; KB/memory; liên tục hội thoại; full-rubric acceptance | Q1/Q5/Q6 thiếu anchor, Q4 chưa có contract/anchor được duyệt để thực thi. Giữ BLOCK full-rubric PASS. |
| Safety/privacy/honesty | NEED_COMPLIANCE và fabricated ACK có quy tắc offline; safety exception cần policy version/owner và review phù hợp | Một phần đã thực thi offline. Chưa có rubric/policy an toàn production được xác nhận; safety calibration NOT_EVALUATED. |
| Router/system actions | Diagnostic route kỳ vọng/thực tế; event trace, human ACK, accepted-silence | Kiểm tra contract offline; chưa phải rubric router/latency hoặc nghiệm thu hành động end-to-end hoàn chỉnh. |
| Evidence/validity | Schema, source identity/hash, quote/span Unicode, completeness và unknown | Validator xác minh tính toàn vẹn và tính verdict theo observation; không tự xác minh nghĩa của observation. |

Q2/Q3 và persona chấm 0–3; thiết kế yêu cầu từng chiều có áp dụng đạt ít nhất 2, không bù bằng trung bình. Persona chưa có gate thực thi. Phân biệt mã persona P1–P6 với các phase triển khai P1–P6.

Validator sở hữu PASS/FAIL/NEEDS_REVIEW/INVALID. Confirmed independent FAIL có precedence hơn unrelated unknown; vẫn giữ review flags. Scoped Q2/Q3 PASS không phải full-rubric PASS; persona không được cộng vào Q3 để nhân đôi trọng số.

Các rubric liên quan:

- Q2/Q3: mimi-q2-q3-annotation-guideline-v0.2.md.
- Persona: mimi-persona-rubric-v0.2.json và bản Markdown cùng tên.
- Q4 draft: mimi-q4-annotation-guideline-v0.1.md và mimi-q4-contract-draft-v0.1.md.
- Quy tắc verdict/evidence/gates: mimi-eval-validator-rules-v0.2.md.
- Quyết định và BLOCK: decision-register-v0.2.json.

Tất cả fixture synthetic còn UNREVIEWED. Semantic accuracy, persona acceptance và production safety đều NOT_EVALUATED. Cấu hình judge Terra chỉ mô phỏng; không gọi provider, truy cập DB hoặc chuyển sang P4.
