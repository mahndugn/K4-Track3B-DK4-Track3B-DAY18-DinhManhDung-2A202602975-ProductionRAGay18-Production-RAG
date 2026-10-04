# Failure Analysis — Lab 18: Production RAG

**Học viên:** Đinh Mạnh Dũng — 2A202602975

## Trạng thái và phạm vi

Đã chạy `main.py` ở chế độ `RAG_OFFLINE=1`: 26 tài liệu có text, 57 baseline chunks, 117 production child chunks, 20 câu hỏi. Hai PDF scan được cảnh báo và bỏ qua. Bộ test cuối: 49 tests pass; lint và `pip check` đạt.

`evaluation_status=unavailable` vì chưa có OPENAI_API_KEY hợp lệ. Các số 0 trong JSON là placeholder, không phải điểm RAGAS. Offline dùng hashing và token overlap; các quan sát dưới đây không chứng minh chất lượng BGE/CrossEncoder/LLM thật.

## RAGAS Scores

| Metric | Naive baseline | Production | Δ |
|---|---|---|---|
| Faithfulness | Chưa đo | Chưa đo | — |
| Answer Relevancy | Chưa đo | Chưa đo | — |
| Context Precision | Chưa đo | Chưa đo | — |
| Context Recall | Chưa đo | Chưa đo | — |

## Bottom-5: trạng thái và phân tích thủ công

Chưa thể xếp hạng bottom-5 theo metrics. `reports/ragas_report.json.failures` giữ 5 câu đầu làm mẫu và đánh dấu `input_order_sample_not_measured_bottom_5`, với worst_metric/score = null. Năm ca dưới đây được chọn thủ công từ lỗi quan sát được; không giả định đây là năm câu có RAGAS thấp nhất.

Bằng chứng: `reports/ragas_report.json.per_question` chứa output/context/ground truth; `reports/retrieval_trace.json` chứa hybrid candidates **trước filter phiên bản** và nguồn context cuối của 5 ca.

### #1 — Câu 4

- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got (trích, toàn văn trong JSON):** [Nguồn: nghi_phep_dac_biet.md] # Chính sách nghỉ phép đặc biệt > Phiên bản: 1.1 | Ngày hiệu lực: 01/03/2024 | Phòng ban: Nhân sự  ## Các trường hợp được nghỉ phép đặc biệt Nhân viên được nghỉ có lương trong các trường hợp sau đây mà không trừ vào phép năm: - **Kết hôn**: 3 ngày làm việc - **Tang lễ** (cha mẹ, vợ/chồng, con): 3 ngày làm việc - **Sin…
- **Context sources:** [Nguồn: nghi_phep_dac_biet.md], [Nguồn: nghi_phep_khong_luong.md], [Nguồn: thu_viec.md]
- **Worst metric:** Chưa đo; hướng chẩn đoán thủ công: Rerank/context selection.
- **Error Tree:** Output trả chính sách nghỉ phép đặc biệt → Context chứa quy định phép năm 15 ngày? Không → Hybrid có nguồn đúng? Có → Fix reranking/context selection.
- **Root cause / observation:** Nguồn nghỉ phép năm v2024 có trong hybrid candidates nhưng bị loại khỏi 3 contexts cuối. Filter phiên bản đã loại nguồn cũ, nhưng lexical reranker vẫn ưu tiên các tài liệu nghỉ phép khác.
- **Suggested fix:** Chạy CrossEncoder thật; benchmark top-k và phrase relevance; bổ sung kiểm thử phân biệt nghỉ phép năm/đặc biệt/không lương.

### #2 — Câu 12

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got (trích, toàn văn trong JSON):** [Nguồn: nghi_phep_khong_luong.md] # Chính sách nghỉ phép không lương > Phiên bản: 1.2 | Ngày hiệu lực: 01/07/2023 | Phòng ban: Nhân sự  ## Điều kiện áp dụng Nhân viên chính thức đã hết phép năm có thể xin nghỉ phép không lương. Thời gian nghỉ phép không lương tối đa là **30 ngày** trong một năm dương lịch. Nghỉ phép không lương không được tính thâm…
- **Context sources:** [Nguồn: nghi_phep_khong_luong.md], [Nguồn: nghi_phep_nam_v2024.md], [Nguồn: nghi_phep_dac_biet.md]
- **Worst metric:** Chưa đo; hướng chẩn đoán thủ công: Retrieval recall + multi-hop.
- **Error Tree:** Output đủ 18 ngày và khoảng lương? Không → Context đủ hai nguồn? Không, thiếu bảng lương → Fix retrieval đa bước trước generation.
- **Root cause / observation:** Nguồn phép năm hiện hành có trong contexts; bang_luong_2024.md không xuất hiện trong 20 hybrid candidates. Fallback chỉ trả context đầu tiên và không tính 15 + 9/3.
- **Suggested fix:** Tách query thành quyền nghỉ phép và lương Senior; union candidates theo subquery rồi rerank; dùng LLM để tổng hợp có trích nguồn.

### #3 — Câu 13

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo giá vì trên 10 triệu.
- **Got (trích, toàn văn trong JSON):** [Nguồn: hoan_chi_dao_tao.md] # Chính sách hoàn chi đào tạo > Phiên bản: 1.1 | Ngày hiệu lực: 01/07/2023 | Phòng ban: Nhân sự & Tài chính  ## Điều kiện được tài trợ Công ty chi trả chi phí đào tạo bên ngoài (chứng chỉ, khóa học dài hạn, hội thảo) cho nhân viên có **thâm niên từ 1 năm trở lên**. Chi phí được tài trợ tối đa **30.000.000 VNĐ/khóa** và …
- **Context sources:** [Nguồn: hoan_chi_dao_tao.md], [Nguồn: dao_tao_noi_bo.md], [Nguồn: nghi_phep_khong_luong.md]
- **Worst metric:** Chưa đo; hướng chẩn đoán thủ công: Reranking / vocabulary gap.
- **Error Tree:** Output trả quy trình laptop? Không → Context có mua_sam.md? Không → Hybrid có nguồn đúng? Có → Fix reranking và phrase coverage.
- **Root cause / observation:** mua_sam.md có trong hybrid candidates nhưng bị loại khỏi contexts. Từ laptop trong câu hỏi khác với thiết bị CNTT/mua sắm trong policy; overlap từ chung nhân viên/triệu làm nguồn đào tạo lên đầu.
- **Suggested fix:** Chạy BGE reranker thật; kiểm tra query expansion laptop → thiết bị CNTT; giữ nguyên ngân sách và ngưỡng phê duyệt khi rewrite.

### #4 — Câu 17

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got (trích, toàn văn trong JSON):** [Nguồn: tam_ung.md] # Chính sách tạm ứng > Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Tài chính  ## Hạn mức tạm ứng Nhân viên có thể tạm ứng tối đa **20.000.000 VNĐ** cho mỗi lần đề xuất. Tạm ứng phải ghi rõ mục đích: công tác, mua sắm, tổ chức sự kiện hoặc chi phí dự án.  ## Thời hạn thanh toán Khoản tạm ứng phải được thanh toán (nộp …
- **Context sources:** [Nguồn: tam_ung.md], [Nguồn: bao_hiem_suc_khoe.md], [Nguồn: nghi_phep_nam_v2024.md]
- **Worst metric:** Chưa đo; hướng chẩn đoán thủ công: Generation / numeric reasoning.
- **Error Tree:** Output có số tiền cụ thể? Không → Context có hạn 15 ngày và phí 2%/tháng? Có → Fix generation/calculation; kiểm tra quy ước pro-rata.
- **Root cause / observation:** tam_ung.md có trong contexts. Fallback trích toàn policy, chưa tính quá hạn 5 ngày và tiền phí. Quy ước tháng 30 ngày cho pro-rata cần được xác nhận, không tự coi là dữ kiện policy.
- **Suggested fix:** Dùng LLM hoặc hàm tính có đơn vị; tách 15 triệu × 2% = 300.000/tháng và pro-rata theo quy ước được phê duyệt; hỏi lại khi thiếu quy tắc.

### #5 — Câu 18

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got (trích, toàn văn trong JSON):** [Nguồn: thu_viec.md] # Chính sách thử việc > Phiên bản: 1.2 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự  ## Thời gian thử việc Thời gian thử việc tiêu chuẩn là **60 ngày** kể từ ngày bắt đầu làm việc. Đối với vị trí quản lý cấp cao (Manager trở lên), thời gian thử việc có thể kéo dài đến 90 ngày.  ## Lương thử việc Nhân viên thử việc được nhận…
- **Context sources:** [Nguồn: thu_viec.md], [Nguồn: bang_luong_2024.md], [Nguồn: thuong_tet.md]
- **Worst metric:** Chưa đo; hướng chẩn đoán thủ công: Generation / source synthesis.
- **Error Tree:** Output có 17 triệu? Không → Context đủ mức lương và tỷ lệ thử việc? Có → Fix generation, không tăng retrieval top-k vô điều kiện.
- **Root cause / observation:** thu_viec.md và bang_luong_2024.md đều có trong contexts; fallback chỉ trả context đầu tiên, chưa kết hợp 85% với mức Junior tối đa 20 triệu.
- **Suggested fix:** Chạy LLM có prompt tính toán và nguồn; xác minh 0,85 × 20.000.000 = 17.000.000; thêm regression numeric sau khi có LLM thật.

## Case study: lương thử việc Junior

Câu 18 cho thấy retrieval đủ nguồn không đảm bảo output hoàn chỉnh. Error Tree đi từ output thiếu số tiền → context có đủ bảng lương và tỷ lệ 85% → lỗi nằm ở generation fallback. Tăng số chunks ở đây chỉ tăng noise; bước cần bổ sung là tổng hợp hai nguồn và tính toán với đơn vị rõ ràng.

## Latency breakdown của lần chạy offline cuối

| Bước | Thời gian (ms) |
|---|---|
| chunking_seconds | 33.277 |
| enrichment_seconds | 3.076 |
| indexing_seconds | 88.325 |
| reranker_loading_seconds | 0.016 |
| Query retrieval_ms, trung bình 20 câu | 2.006 |
| Query rerank_ms, trung bình 20 câu | 0.343 |
| Query generation_ms, trung bình 20 câu | 0.000 |

Các số này là smoke-test latency trên máy hiện tại, sau khi dependencies đã được nạp/cache. Không dùng để ước lượng latency model pretrained hoặc API.

## Nếu có thêm 1 giờ

1. Chạy model pretrained và RAGAS thật với key hợp lệ; giữ cùng test set/model answer cho baseline và production.
2. Ablation hybrid top-k, reranker và enrichment trên các ca 4/12/13; đo retrieval hit@k theo nguồn ground truth.
3. Bổ sung query decomposition cho multi-hop và kiểm tra số học ở generation; xác nhận quy tắc tính pro-rata.
4. Khi metrics completed, thay phần chọn ca thủ công bằng bottom-5 thực từ `failure_analysis()` rồi đối chiếu Error Tree.
