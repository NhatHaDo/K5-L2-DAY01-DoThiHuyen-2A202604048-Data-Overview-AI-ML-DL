# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 07/11/2026

**Runtime Colab:** GPU (T4, `Device: cuda`)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.14.1+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh**. Record không có `bbox_xyxy` hay `polygon_xy`, tức là không chỉ ra vị trí nào trong ảnh; mỗi ảnh chỉ có một danh sách top-5 xếp hạng (cab, minibus, police_van, recreational_vehicle, streetcar). Ảnh `traffic` có hàng chục xe buýt, ô tô và người, nhưng output vẫn chỉ là một nhãn cho toàn ảnh.

3. Class list đi kèm taxonomy của checkpoint: `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K (1000 lớp). Model chỉ xếp hạng trong danh sách cố định này, không tự nghĩ ra tên lớp. Ví dụ ImageNet-1K không có lớp "cảnh đường phố" hay "giao thông", nên model buộc phải chọn một loại xe cụ thể (`cab`).

4. Cùng một vật có thể có tên/mã khác nhau ở các taxonomy khác nhau. Ví dụ trong lab này, xe buýt ở ImageNet-1K có các lớp như `minibus` (id 654), còn ở COCO-80 là `bus` (id 5). `class_id` là mã máy dùng khi huấn luyện, `class_name` để người đọc, `taxonomy_name` cho biết mã thuộc bộ lớp nào. Thiếu một trong ba thì không đối chiếu hay gộp dữ liệu chính xác được.

5. Guideline cần quy định: chọn lớp theo vật/chủ đề **chính** của ảnh (vd. vật chiếm diện tích lớn nhất hoặc ở trọng tâm), thứ tự ưu tiên khi nhiều vật ngang nhau, và khi nào đánh dấu "không rõ"/escalation thay vì đoán. Với `traffic`, phải nói rõ gán theo vật nổi bật nhất (xe buýt xanh ở tiền cảnh) hay theo loại xe chiếm số lượng nhiều nhất.

6. `score` chỉ là mức "tin" của model, dùng để xếp hạng/lọc, và model có thể sai mà score vẫn cao. Ở đây top-1 là `cab` (0.51) trong khi vật nổi bật nhất trong ảnh là xe buýt; detection trên cùng ảnh còn trả về `bus` với score 0.91. Nếu lấy prediction làm ground truth thì lỗi của model sẽ bị đưa ngược vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Record: `class_name`=person ; `score`=0.912624 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Tọa độ tính bằng pixel, gốc (0, 0) ở góc trên-trái ảnh (ảnh `kitchen` rộng 640 px, cao 427 px), x tăng sang phải, y tăng xuống dưới. Box này bao người đầu bếp đứng quay lưng ở giữa-phải ảnh: góc trên-trái ở (385.33, 69.24), tức cách mép trái khoảng 385 px và cách mép trên khoảng 69 px (ngang đỉnh đầu); góc dưới-phải ở (498.92, 348.92). Box rộng 498.92 − 385.33 = 113.58 px và cao 348.92 − 69.24 = 279.68 px — cao gấp khoảng 2.5 lần chiều rộng, đúng dáng một người đứng.

3. Ngưỡng `0.20`: 17 prediction (thêm spoon ×3, potted plant, dining table, bottle…). Ngưỡng `0.60`: 6 prediction (person ×2, bowl ×2, oven ×2). Ngưỡng `0.35` dùng cho JSON: 11 prediction.

4. Hạ ngưỡng từ 0.60 xuống 0.20 thì số box tăng gần gấp ba (6 → 17), chủ yếu là box score thấp cho vật nhỏ, bị che hoặc lẫn với vật khác (spoon, bottle, dining table). Reviewer phải xem từng box thêm vào: xóa box sai class hoặc trùng, chỉnh lại biên box, và kiểm tra vật bị bỏ sót. Đổi lại recall cao hơn là khối lượng kiểm tra lớn hơn.

5. Quy tắc box: Box phải ôm sát phần **nhìn thấy được** của vật — mỗi cạnh chạm điểm ngoài cùng của vật, không chừa nền thừa, không cắt mất phần vật; mỗi instance một box riêng, không gộp nhiều vật cùng lớp vào một box.

6. Vật bị cắt mép: ở mép trái ảnh `kitchen` chỉ thấy một bàn tay/cánh tay thò vào cạnh bàn bột, phần thân người nằm ngoài khung ảnh; danh sách ở ngưỡng 0.60 có 2 `person` dù trong khung chỉ có một người đứng, nên box `person` thứ hai nhiều khả năng là bàn tay này. Xử lý bằng **guideline**, vì vật cắt mép và bị che lặp lại ở rất nhiều ảnh nên cần quy tắc chung: "chỉ vẽ box cho phần nhìn thấy và gắn cờ `truncated`", kèm ngưỡng tối thiểu (vd. thấy được phần nào thì mới gán `person`). Chỉ escalation khi không xác định được đó là vật gì.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 3 điểm đầu của `polygon_xy`=[446.0, 70.0], [445.0, 71.0], [444.0, 71.0]

2. Box chỉ là hình chữ nhật bao ngoài, nên với người đầu bếp (box detection rộng 113.58 × cao 279.68 px) nó chứa cả tường và xoong chảo phía sau. Polygon của cùng người này có 348 điểm, bắt đầu từ đỉnh đầu (~446, 70). Polygon đi theo đường biên thật của vật (đầu, vai, tay, tạp dề), cho biết hình dạng, phần lõm và pixel nào thuộc vật thể; nhờ đó tách được các vật nằm sát nhau mà box của chúng chồng lên nhau.

3. `instance_id` (dạng `kitchen-001`) chỉ đánh số để phân biệt từng instance trong bộ output này, vd. hai `bowl` cùng lớp sẽ có hai `instance_id` khác nhau. Nó không phải class ID vì không cho biết lớp gì (lớp nằm ở `class_id`/`class_name`), và không phải tracking ID vì không nối cùng một vật qua nhiều frame/ảnh; chạy lại hoặc sang ảnh khác thì số này không còn ý nghĩa.

4. Quy tắc vẽ biên mask: Polygon đi sát biên ngoài của phần vật nhìn thấy được, không bao vùng nền; phần bị vật khác che thì không vẽ vào; mỗi pixel chỉ thuộc tối đa một instance — hai mask không được chồng lên nhau.

5. Chỗ khó: box `spoon 0.49` ở góc trên-phải bao một cụm muôi/vá treo sát nhau trên tường, các cán chạm và chồng lên nhau nên khó tách biên từng cái; ngoài ra muôi/vá không có lớp riêng trong COCO-80. Phần tách biên giữa hai vật chạm nhau xử lý bằng **guideline** (chia biên theo đường tiếp giáp nhìn thấy được, mỗi vật một mask). Câu hỏi "muôi có tính là `spoon` không" là vấn đề taxonomy nên phải **escalation** cho Lab Coach, vì annotator tự quyết sẽ gán không thống nhất giữa các người.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp (`class_id` + `class_name` + `taxonomy_name`) cho cả ảnh | `traffic`: top-1 `cab` 0.51, `minibus` 0.16; ảnh chủ yếu là xe buýt nhưng ImageNet-1K không có lớp "cảnh giao thông" → khó chọn một lớp | Chọn một lớp chính theo guideline; ca mơ hồ thì escalation | Lớp có đúng guideline không, có thống nhất giữa các annotator không |
| Phát hiện vật thể | Mỗi vật một box `[x_min, y_min, x_max, y_max]` (pixel) + lớp | `kitchen`: 6 box ở ngưỡng 0.60 nhưng 17 box ở 0.20; có 2 `person` trong khi người thứ hai chỉ lộ bàn tay ở mép trái; hàng chục xoong chảo treo tường không có box vì COCO-80 không có lớp nồi/chảo | Vẽ box ôm sát từng vật nhìn thấy, chọn đúng lớp, gắn cờ cắt mép/bị che, không bỏ sót | Box sót, box thừa/trùng, box lệch biên, sai class |
| Instance segmentation | Mỗi instance một polygon/mask (pixel) + lớp | `kitchen`: `spoon 0.49` bao cả cụm muôi treo chạm nhau; `bowl 0.37` là vật nhỏ, score thấp | Vẽ polygon theo biên vật, tách riêng các instance chạm nhau | Biên mask có sát không, mask chồng nhau, instance bị gộp hoặc sót |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai đã được notebook cố định và kiểm tra checksum; không đưa ảnh cá nhân, khuôn mặt, biển số, dữ liệu nội bộ hay thông tin định danh vào Colab, báo cáo hoặc repo công khai.

2. Dừng ngay, không tải lên hay xử lý tiếp, và báo cho Lab Coach (hoặc GV/Lab Lead).

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
