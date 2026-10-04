# Reflection — Lab 18: Production RAG

**Họ tên:** Đinh Mạnh Dũng
**MSSV:** 2A202602975
**Khóa:** K4 – Track 3B
**Ngày tổng kết kỹ thuật:** 04/10/2026

Bản tổng kết này ghi lại việc triển khai và kiểm tra trong repo. Kế hoạch project ở phần 3
là đề xuất cho trợ lý chính sách nội bộ; không khẳng định đây là project cá nhân đã triển khai.

## Phần 1: Mapping bài giảng

| Concept | Module / hàm | Observation |
|---|---|---|
| Semantic chunking | M1 `chunk_semantic()` | Encode câu bằng all-MiniLM-L6-v2; tách khi cosine giữa hai câu liền kề dưới threshold. Model được cache. Trong offline, hashing chỉ kiểm tra luồng, không chứng minh chất lượng ngữ nghĩa. |
| Hierarchical chunking | M1 `chunk_hierarchical()`, pipeline `run_query()` | Parent tối đa 2048 ký tự, child 256. ID chứa hash nguồn và nội dung để không trùng giữa các tài liệu. Retrieve/rerank child rồi trả parent; loại trùng parent trước khi chọn 3 contexts. |
| Structure-aware | M1 `chunk_structure_aware()` | Tách theo heading Markdown, giữ bảng/list trong section; heading trong fenced code không tạo section mới. |
| BM25 + Dense fusion | M2 `segment_vietnamese()`, `DenseSearch`, `reciprocal_rank_fusion()` | Normalize tiếng Việt; bỏ underscore, cùng tokenizer cho query/corpus. BGE-M3 và BM25 kết hợp bằng tổng 1/(60+rank+1), không cộng trực tiếp hai thang điểm khác nhau. |
| Cross-encoder | M3 `CrossEncoderReranker.rerank()` | Predict cặp query–child gốc, giữ score cũ và score mới. Không đưa HyQA/summary sinh ra vào evidence của câu trả lời. Offline dùng token overlap và không được coi là benchmark BGE. |
| RAGAS | M4 `evaluate_ragas()`, `failure_analysis()` | Có bốn metrics và records từng câu; lỗi/thiếu key được đánh dấu unavailable. Không thể kết luận metric thấp nhất hay mức cải thiện khi chưa có điểm thật. |
| Contextual embeddings | M5 `_enrich_single_call()`, `enrich_chunks()` | Một request JSON/chunk; summary, HyQA và context được index cùng văn bản. Khi không có key, dùng extractive fallback. Generated metadata không được ghi đè source/parent ID. |

Latency build và từng query được lưu trong `reports/ragas_report.json` →
`aggregate.runtime`. Các số đo offline chỉ mô tả chi phí chạy smoke test trên máy này.

## Phần 2: Khó khăn và giải quyết

1. **Thiếu dependencies:** `ModuleNotFoundError: No module named 'dotenv'`.
   Kiểm tra interpreter và `pip --version`; môi trường ảo ban đầu gần như trống.
2. **Python 3.13 với dependencies cũ:** `error: metadata-generation-failed` và
   `ERROR: Unknown compiler(s)` khi build NumPy 1.26.4.
   Nguyên nhân là LangChain 0.2 yêu cầu NumPy 1.x, trong khi Python 3.13 không có wheel
   phù hợp cho NumPy 1.26. Giải pháp cuối cùng: nâng RAGAS lên 0.2.15, LangChain lên 0.3,
   dùng interpreter Python 3.13 hiện có; kiểm tra adapter với dữ liệu giả lập.
3. **Interpreter tải thêm bị Windows chặn:**
   `ImportError: DLL load failed while importing _ssl: An Application Control policy has blocked this file.`
   Không thay đổi policy Windows. Tiếp tục dùng interpreter hiện có import SSL thành công.
4. **Không có API key thật:** `.env` chứa key mẫu. Nhận diện placeholder trong config,
   không gửi request với key mẫu; lưu `evaluation_status=unavailable` và giữ records
   để phân tích thủ công. Cần chạy lại với key hợp lệ trước khi báo cáo điểm RAGAS.
5. **Lỗi logic scaffold:** parent ID theo index có nguy cơ trùng giữa documents và pipeline
   chỉ dùng child làm context. Dùng hash nguồn/nội dung, thêm parent lookup và regression test
   để xác nhận context trả về parent, không chứa synthetic enrichment.

Kiến thức cần bổ sung: calibrate threshold cho tiếng Việt, đánh giá retrieval với corpus phiên bản,
đọc schema RAGAS 0.2 và đo latency model thật. Unit test adapter không thay thế việc chạy model/API thật.

## Phần 3: Action plan đề xuất

### Project: Trợ lý tra cứu chính sách nhân sự và CNTT nội bộ

Hiện tại repo có corpus Markdown/PDF, 20 câu hỏi ground truth, baseline dense-only và
pipeline hierarchical → enrichment → hybrid → rerank → answer → evaluation.
Các giới hạn còn lại: PDF scan cần OCR; không có RAGAS thật; offline answer là trích nguyên context,
chưa tổng hợp multi-hop hay tính toán số học. Metadata phiên bản dùng tiêu đề và số phiên bản
trong tài liệu, cần quy trình quản lý phiên bản rõ ràng nếu triển khai thực tế.

### Plan áp dụng

1. Chunking: dùng hierarchical cho chính sách dài, structure-aware cho tài liệu có bảng;
   so sánh recall trước khi chọn threshold semantic tiếng Việt.
2. Search: giữ hybrid RRF, lập benchmark version conflict, phủ định và câu hỏi đa tài liệu.
   Đối chiếu filter phiên bản hiện hành với truy vấn yêu cầu lịch sử.
3. Reranking: chạy BGE reranker trên CPU/GPU thực tế; so sánh top-k 10/20 và latency p50/p95.
4. Evaluation: chạy baseline và production với cùng model/key/test set; bổ sung retrieval hit@k,
   phân loại lỗi, kiểm tra tính toán và negation. Chỉ dùng scores khi status completed.
5. Enrichment: thử combined/contextual; cache theo hash nội dung và phiên bản model,
   đo chi phí và lợi ích HyQA trước khi áp dụng toàn corpus.

### Timeline

- Tuần 1: thêm OCR và metadata nguồn/ngày hiệu lực; chạy model thật và thu baseline RAGAS;
  rà soát ground truth cùng người quản lý chính sách.
- Tuần 2: ablation chunking/hybrid/rerank/enrichment; bổ sung câu hỏi multi-hop/numeric;
  chọn cấu hình theo chất lượng, latency và chi phí; chạy lại Error Tree bottom-5.

### Tiêu chí hoàn thành

Ít nhất 3 metrics RAGAS ≥ 0.70 trên bộ kiểm thử độc lập; không trả chính sách đã bị thay thế
cho câu hỏi hiện hành; trả lời có nguồn và giữ nguyên phủ định. Mục tiêu latency đặt sau khi
đã có benchmark phần cứng thực tế, không suy ra từ smoke test.

## Kết quả kiểm tra đã ghi nhận

- `RAG_OFFLINE=1 python -m pytest tests/ -q`: **49 passed**, 10,53 giây ở lần chạy xác nhận.
- `RAG_OFFLINE=1 python main.py`: exit 0; 26 tài liệu text, 57 baseline chunks và 117 production children; đủ 20 records mỗi báo cáo.
- Lint và dependency check đạt. RAGAS thật chưa chạy; API key mẫu, không có scores chất lượng để so sánh.
- Traceback chẩn đoán lần nạp đầu cho thấy chờ ở `torch.__init__._load_dll_libraries` và import `pyarrow.dataset`. Offline segmentation được giữ nhẹ, không import pipeline translation của underthesea; nhánh production vẫn dùng underthesea.
- Các ca 4/12/13 cho thấy lỗi retrieval/rerank; ca 17/18 cần tính toán/tổng hợp. Chi tiết và bằng chứng nằm trong failure analysis và retrieval trace.
