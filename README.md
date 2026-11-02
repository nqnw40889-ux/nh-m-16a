## BẢN ĐÓNG GÓI ĐÃ CHẠY

Gói này đã có mô hình + metadata + reports từ CSV tải trực tiếp UCI; có thể đưa lên GitHub ngay. Không cần chạy export_colab nếu dùng gói này.

- Môi trường huấn luyện gói đi kèm: **Python 3.12**, requirements đã ghim phiên bản. Khi tạo bản triển khai mới chọn Advanced settings → Python 3.12. Nếu giữ website cũ làm dự phòng, triển khai gói này trong repository/app mới.
- Kết quả giữ nguyên LightGBM nền, ngưỡng 0.266037. Test: TP 38, FN 13, FP 16, TN 1433. F1 0.723810, AP 0.800627.
- Tuning LightGBM 12×3 fold đã chạy; AP validation 0.743189 thấp hơn cấu hình nền 0.783474 nên không được chọn.
- Sigmoid CV đã chạy; Brier validation 0.014158 so với gốc 0.013170 (thấp hơn tốt hơn), nên không chọn hiệu chỉnh. UI sẽ ghi chính xác chưa hiệu chỉnh.
- Nguồn CSV: https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip
- CSV không kèm trong gói, dùng bản tải từ UCI hoặc CSV của nhóm để tái lập.
- Phần dưới hướng dẫn thêm cách sử dụng mô hình gốc Colab nếu bạn muốn giữ đúng tệp mô hình riêng.

# NHÓM 16 — Machine Health Studio

Ứng dụng Streamlit cho demo phân loại hỏng hóc AI4I 2020. Không kết nối máy thật, không xác định thời gian đến hỏng. Không kèm mô hình/dữ liệu giả. Cần mô hình hiện có của nhóm hoặc chạy train.py với CSV gốc.

## A. Nâng cấp từ notebook Colab hiện tại (khuyến nghị)

1. Giữ nguyên notebook và website cũ để có bản dự phòng.
2. Tải ZIP mã nâng cấp vào máy. Trong Colab đang chứa các biến sau huấn luyện, tạo một ô:

```python
from google.colab import files
files.upload()  # Chọn Nhom16_Upgrade.zip
```

3. Tạo ô mới chạy:

```python
import zipfile, sys
with zipfile.ZipFile('Nhom16_Upgrade.zip') as z:
    z.extractall('nhom16_upgrade')
sys.path.insert(0, '/content/nhom16_upgrade')
exec(open('/content/nhom16_upgrade/export_colab.py', encoding='utf-8').read())
```

Nếu trình duyệt đổi tên tệp tải lên, sửa lại tên ZIP trong mã cho đúng.
Cần các biến pipeline_final, nguong_tot_nhat, ten_tot_nhat, X_train/y_train, X_val/y_val, X_test/y_test từ notebook cũ. Nếu mất phiên Colab, chạy lại các ô cũ hoặc khôi phục từ tệp đã lưu; không nhập số liệu bằng tay để giả lập.

4. Tải và giải nén `Nhom16_Web_Upgrade.zip` được tạo. Gói này chứa mô hình gốc giữ nguyên, metadata về phạm vi train, phiên bản thư viện và báo cáo thật. Không tự huấn luyện hay hiệu chỉnh ở bước này.
5. Trên repository GitHub của website, thay app.py và requirements.txt, thêm core.py, evaluation.py, metadata.json và thư mục reports; giữ mo_hinh_demo.joblib do gói mới xuất. Đưa các tệp ở thư mục gốc, không tải nguyên ZIP. Có thể tải cả các tệp huấn luyện/hướng dẫn để lưu mã nguồn; không cần đưa CSV dữ liệu người dùng lên repository.
6. Website Streamlit nhận bản cập nhật. Dùng Python 3.13 khớp Colab hiện tại. Không tự đổi phiên bản scikit-learn trong requirements của mô hình cũ. Trước thuyết trình mở website, kiểm tra cả hai mẫu và các trang.

## B. Các chức năng đã có trong mã

- Kiểm tra đủ sáu cột, nhãn Type, số hữu hạn, giá trị âm, số nguyên cho rpm/min.
- Cảnh báo từng biến ngoài min/max **train**, mặc định không dự đoán; người dùng có thể xác nhận thử nghiệm ngoài phạm vi. Đây không phải kiểm định mọi quan hệ vật lý hoặc bảo đảm đúng trong phạm vi.
- Dự đoán một mẫu; hai mẫu validation được chọn đúng chỉ dùng minh họa. Không tuyên bố đó là kiểm thử độc lập.
- CSV tối đa 10 MB/10.000 dòng: kiểm tra từng dòng, không loại bỏ âm thầm lỗi, tải kết quả CSV, không dùng cột nhãn đầu vào.
- Lịch sử tối đa 1.000 dòng/phiên, tải CSV và xóa. Không lưu bền vững; tải trước khi đóng phiên.
- TreeSHAP native LightGBM giải thích log-odds. Với hiệu chỉnh, giải thích mô hình nền; không gọi đó là đóng góp trực tiếp vào xác suất đã hiệu chỉnh hoặc nguyên nhân vật lý.
- Trang đánh giá hiển thị bảng thật, PR, AP và PR-AUC hình thang riêng, train–validation, confusion matrix. Không có tệp thì không có số liệu thay thế.
- README, requirements phiên bản chính xác, train split indices và mã kiểm thử.

## C. Huấn luyện nâng cao (tùy chọn, thực nghiệm mới)

Đã bổ sung mã chạy nhưng không thể tuyên bố kết quả trước khi thực thi:

```bash
python train.py --csv ai4i2020.csv --output trained_run --advanced
```

Hoặc Colab (sau khi cài requirements, từ thư mục mã):

```python
!python /content/nhom16_upgrade/train.py --csv /content/ai4i2020.csv --output /content/nhom16_advanced --advanced
```

- Chia 70/15/15 phân tầng, seed 42 như notebook gốc.
- Chạy 8 cấu hình nền. Tinh chỉnh **LightGBM** với 12 cấu hình ngẫu nhiên x 3 fold trên train, scoring AP. Không tuyên bố đã tối ưu cả 4 thuật toán.
- Bộ tiền xử lý nằm trong pipeline, được fit riêng từng fold.
- Chọn mô hình theo AP validation. Thử hiệu chỉnh sigmoid bằng 3 fold train-only, chọn giữa có/không hiệu chỉnh theo Brier validation. Hiệu chỉnh không bảo đảm tốt hơn và không biến dữ liệu mô phỏng thành rủi ro máy thật.
- Ngưỡng F1 validation, nếu hòa chọn ngưỡng cao nhất. Cố định trước khi tính test. Không refit lên train+validation sau chọn ngưỡng.
- Báo cáo metrics, PR, calibration validation, CV results, split indices, SHA256 CSV, model và requirements được ghi vào output. Cần chép model/metadata/reports/requirements từ output sang thư mục ứng dụng; giữ app.py/core.py/evaluation.py.
- **Tập test cũ đã được nhóm xem.** Các lần mở rộng dùng lại nó là đánh giá thăm dò trên holdout đã quan sát, không phải một kiểm định mới hoàn toàn độc lập. Không chọn phương pháp dựa trên việc test tăng hay giảm. Muốn kết luận xác nhận mới cần thêm dữ liệu chưa quan sát.
- Sau khi đã có output, lệnh dừng để tránh ghi đè; chọn output mới cho thực nghiệm mới.

## D. Chạy web trên máy

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Chỉ nạp tệp joblib do nhóm tự tạo và tin cậy. Không có chức năng tải pickle/joblib từ người xem web.

## E. Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Kiểm thử tự động dùng dữ liệu nhân tạo **chỉ kiểm tra chức năng phần mềm**, không đánh giá chất lượng AI4I. Kiểm thử nghiệm thu website: mẫu thường/mẫu hỏng; Type sai trong CSV; thiếu cột; số thiếu/NaN/inf; giá trị âm; ngoài phạm vi có/không cho phép; lô trộn hợp lệ/lỗi; tải kết quả; lịch sử/xóa; mở lại bằng mô hình đã lưu. Chụp màn hình và ghi kết quả thật, không ghi "pass" khi chưa chạy.

## F. Các điểm cần ghi trong báo cáo

Nguồn UCI: https://archive.ics.uci.edu/dataset/601/ai4i
Tách số liệu notebook gốc và thực nghiệm nâng cao. AP khác PR-AUC hình thang. Ngưỡng được chọn bằng validation, không phải ngưỡng kỹ thuật của nhà sản xuất. Kết quả không hỏng không xác nhận an toàn. Hiệu quả test không chứng minh triển khai thực tế.

Nguồn kỹ thuật: https://scikit-learn.org/stable/modules/calibration.html và https://lightgbm.readthedocs.io/en/latest/Parameters.html (pred_contrib).
