> Historical student-lens-r1 design note. A later separate persona draft now exists at mimi-persona-rubric-v0.2.md and ../fixtures/mimi-persona-suite-draft-v0.2.json. Persona acceptance remains NOT_EVALUATED; no persona scoring or live run has been added.

# Student lens r1 — Q2/Q3 và câu “em phải làm sao?”

Nguồn thay đổi: product owner yêu cầu đưa góc nhìn nhập vai học sinh vào rubric và bổ sung cặp than/xin giải pháp. Đây là ngữ liệu synthetic để thiết kế; không phải phỏng vấn học sinh thật, không đại diện cho mọi học sinh Việt Nam và không phải persona chuẩn của Mimi. Toàn bộ fixture mới vẫn UNREVIEWED.

## Hai câu hỏi chấm độc lập

- Q2: Mimi có bắt đúng điều học sinh cần ở lượt này không? Nếu cần hỗ trợ thực tế, học sinh có biết bước bắt đầu chưa? Có bị buộc phải lặp lại ranh giới hoặc trả lời thêm không cần thiết không?
- Q3: Cách nói có tự nhiên, tôn trọng và hợp độ tuổi/bối cảnh không? Có thúc ép, xem thường, dài gây quá tải hoặc lặp khuôn mẫu không?

Một câu nói trung tính nhưng tự nhiên vẫn có thể Q3=3. Không cộng điểm vì dùng “tui/tớ”, tiếng lóng, emoji hoặc bắt chước cách nói của một học sinh giả lập. Cũng không mặc định cách nói trung tính là persona đạt chuẩn. Nét nhận diện Mimi, vai trò/ranh giới quan hệ và nhất quán giữa các chủ đề cần một bộ persona cases riêng với tiêu chí được duyệt. Q3 đánh giá phần phong thái chung; full persona acceptance vẫn NOT_EVALUATED. Revision này chưa xây thêm bộ persona đó hoặc lane mới.

## Cặp đối chứng 1: giữ nguyên câu cảm thán và hai phản hồi

Bối cảnh chung: “Mai kiểm tra Toán. Em ôn mấy ngày rồi mà vẫn hay nhầm dạng bài.”

Câu cuối giống hệt trong cả hai nhánh: **“Trời ơi, em phải làm sao đây...”**

| Nhánh | Tin nhắn trước đó | Nhu cầu được giả định trong fixture draft |
|---|---|---|
| T21 | “Đầu em đang rối. Em biết phải xem lại bài sai rồi, giờ chỉ muốn than tí, đừng khuyên nữa nha.” | Than, ranh giới chỉ nghe vẫn còn hiệu lực |
| T22 | “Tối nay em còn 45 phút. Em đang cần một cách bắt đầu ôn ngay, chỉ em với.” | Xin giải pháp rõ, dù câu cuối có cảm thán |

| Cùng phản hồi của Mimi | T21: đang than | T22: xin giải pháp |
|---|---|---|
| “Ôn mấy ngày mà vẫn nhầm thì nản thật. Mimi đang nghe em.” | C1: Q2=3, Q3=3 | C2: Q2=0, Q3=3; cả lượt không hỗ trợ yêu cầu rõ |
| “Em chọn một dạng hay nhầm nhất, làm lại một bài trước rồi đánh dấu bước bị sai nhé.” | C2: Q2=0/NEED_COMPLIANCE khi đủ bằng chứng | C1: Q2=3, Q3=3 |

C3 của T21 dùng câu phản chiếu trung tính, vẫn Q3=3 trong giả định draft. C3 của T22 giữ lời khuyên hữu ích nhưng thêm “Có thế mà cũng không tự nghĩ ra à?”: Q2 vẫn phản ánh đúng hành động hỗ trợ, Q3=0 vì coi thường. Nhờ vậy test không đồng nhất Q2 với Q3, cũng không dùng slang làm thước đo persona.

## Cặp đối chứng 2: giữ thêm ranh giới “đừng hỏi thêm”

T23/T24 cũng dùng cùng bối cảnh, câu cuối và hai phản hồi ở trên. Cả hai đều có explicit_no_questions=true.

- T23: “Đừng hỏi thêm nha. Mẹ đã chia lịch học rồi. Em chỉ muốn than chút, không muốn nghe thêm cách học nữa.” → chỉ nghe; không tự khuyên hoặc hỏi tiếp.
- T24: “Đừng hỏi thêm nha, cho em một bước để bắt đầu tối nay. Em còn 45 phút trước khi ngủ.” → giúp trực tiếp, không hỏi xin phép thêm.

T23-C3 hỏi thông tin (clarify) và T24-C3 hỏi “Em có muốn thử làm lại một bài hay nhầm nhất trước không?” (advise) đều đi ngược ranh giới không hỏi trong ngữ cảnh này: Q2=0/QUALITY. Không tự áp mandatory NEED_COMPLIANCE chỉ vì có câu hỏi; mandatory đó dành cho lời khuyên không cần thiết khi listen-only còn hiệu lực và đã đủ căn cứ.

## T25: không bịa độ chắc chắn

Prefix chỉ cho biết học sinh đang ngồi nhìn vở và nói cùng câu cảm thán. Lịch sử được cung cấp đầy đủ, nhưng mục đích vẫn chưa đủ rõ. Cả ba giả định phản hồi reflect/clarify/advise đều mang advice_requested=unknown, Q2 certainty=unknown và một material review flag. Kết quả harness là NEEDS_REVIEW, không INVALID. Không coi mọi câu có “làm sao” là xin giải pháp, cũng không coi mọi câu cảm thán là đang than. Các đoạn prefix được thiết kế rõ ở T21–T24 vẫn cần người chấm xác minh giả định diễn giải trước khi thành gold.

## Điều chỉnh liên quan ở fixture cũ

- T05-C1/C2 có thêm một tin nhắn thay đổi nhu cầu rõ ràng trước câu “Giờ em phải làm sao?”, tránh vô tình biến case thành kiểm từ khóa. T05-C3 giữ listen-only cũ khi học sinh chưa thay đổi rõ, dù nhu cầu hiện tại còn nhập nhằng.
- T05-C2 và T08-C3 thống nhất với T02-C2: yêu cầu giải pháp rõ mà cả lượt chỉ phản chiếu → Q2=0 theo anchor hiện có. Hỏi xin phép thêm hoặc hỗ trợ bị trì hoãn có thể thuộc Q2=1; phản chiếu ngắn rồi giúp thực chất không bị xem là bỏ qua nhu cầu.
- T08 không còn liệt kê clarify là allowed khi đã cấm hỏi. No-questions áp cả câu hỏi có lời khuyên bên trong, không chỉ action clarify.
- Chỉ cập nhật giả định fixture synthetic và binding của template còn rỗng; không sửa human adjudication hoặc attachment gốc.

## Kiểm chứng và giới hạn

Suite hiện có 25 nuclei × 3 candidates + 4 controls = 79 case. T21–T25 cùng family/split để tránh tách các biến thể cùng mục đích sang hai tập. Kết quả quan sát được ở reports/student-lens-r1/; các receipt 64 case/108 tests trước đó được giữ như lịch sử của revision trước, không áp sang nguồn mới.

Test xác minh cặp text được giữ cố định, mapping scripted labels/scores, source integrity và quyết định validator. Không có LLM nào thật sự phân loại câu nói trong lần chạy offline. Semantic agreement, false PASS/FAIL, safety calibration và persona acceptance đều NOT_EVALUATED. Không sửa generator, không gọi provider/DB, không chuyển sang P4.
