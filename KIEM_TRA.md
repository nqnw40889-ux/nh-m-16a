# KIỂM TRA BẢN NÂNG CẤP

- 7 kiểm thử tự động đạt: dữ liệu sai/thiếu; ngoài phạm vi có và không cho phép; lô trộn lỗi/hợp lệ; bảo toàn chỉ số; lưu/nạp; luồng UI dự đoán và lịch sử.
- Đã chạy AppTest với mô hình AI4I thực: năm trang không phát sinh exception; hai mẫu validation trả đúng 0/1; TreeSHAP trả bảy đóng góp hữu hạn.
- Đã chạy toàn bộ huấn luyện, CV và calibration trên CSV UCI. Kết quả trong reports là đầu ra thật.
- Chưa kiểm thử trực quan bằng trình duyệt trên Streamlit Cloud, chưa triển khai vào repository của người dùng. Cần kiểm tra sau khi tải lên.
- Test đã được quan sát ở thực nghiệm trước; bản mở rộng được ghi rõ là thăm dò, không phải kiểm định xác nhận độc lập mới.
