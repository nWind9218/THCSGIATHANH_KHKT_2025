# Persona draft r1 — kiểm chứng offline

Hướng bốn sắc thái đã được user xác nhận. Sáu chiều P1–P6 và các khoảng điểm minh họa còn PROPOSED_FOR_PRODUCT_REVIEW. Persona acceptance, human agreement và semantic calibration: NOT_EVALUATED.

- 17 tình huống / 51 phản hồi synthetic; mọi human annotation UNREVIEWED.
- 146 kiểm tra offline đạt (cấu trúc/negative schema, source/hash/span, coverage, template rỗng và regression).
- 79 case behavior cũ vẫn khớp toàn bộ scripted expectations; không có persona scoring được lồng vào đó.
- Không đổi generator, validator, observer contract Q2/Q3, không gọi provider/DB hoặc chuyển sang P4.

Lệnh đã chạy:

```powershell
.\.venv-local\Scripts\python.exe -m pytest -q tests/eval_v02 --junitxml=mimi_eval/reports/persona-draft-r1/pytest-final.xml
.\.venv-local\Scripts\python.exe -m mimi_eval.replay --offline --validate-only
```

Bản rubric để review: ../../specs/mimi-persona-rubric-v0.2.md. JSON là nguồn anchor, Markdown trình bày cùng nội dung. Schema persona chỉ là schema cho fixture draft, không phải observer output/API contract.

Nguồn và kết quả: source-manifest-before.json, source-manifest-after.json, pytest-final.xml và verification.json. Không ghi đè receipt của student-lens-r1 hoặc baseline cũ. Rollback riêng phần draft bằng revert commit bổ sung này, giữ framework Q2/Q3 trước đó; không có dữ liệu production hay migration để phục hồi.

Sau khi cùng review anchor: xây observation contract persona có version riêng và hiệu chỉnh với human gold. Không tự đưa draft thành acceptance gate hoặc đổi model sản phẩm.
