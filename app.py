from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import pandas as pd
import streamlit as st
from core import FEATURES, NUM, LABELS, predict_batch, contributions

ROOT = Path(__file__).parent
st.set_page_config(page_title='Nhóm 16 · Machine Health', page_icon='⚙️', layout='wide')
st.markdown('''<style>
.stApp {background:#f4f7fb} .block-container {max-width:1250px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-.025em;color:#142941}
div[data-testid="stMetric"] {background:white;border:1px solid #e1e8f0;border-radius:16px;padding:18px}
[data-testid="stSidebar"] {background:#12263d} [data-testid="stSidebar"] * {color:#e3edf7}
.stButton>button[kind="primary"],.stFormSubmitButton>button {border-radius:10px}
div[data-testid="stForm"] {background:white;border-radius:18px;border:1px solid #e1e8f0;padding:24px}
</style>''', unsafe_allow_html=True)
st.sidebar.title('⚙️ MACHINE HEALTH')
st.sidebar.caption('NHÓM 16 · HỌC MÁY')
page = st.sidebar.radio('Chức năng', ['Dự đoán', 'CSV hàng loạt', 'Lịch sử', 'Đánh giá', 'Thông tin'])
st.sidebar.divider()
st.sidebar.caption('Phân loại trên AI4I 2020 mô phỏng. Chưa kết nối cảm biến thực tế.')

@st.cache_resource
def load_bundle(stamp):
    return joblib.load(ROOT / 'mo_hinh_demo.joblib')

path = ROOT / 'mo_hinh_demo.joblib'
if not path.exists():
    st.title('Thiết lập ứng dụng')
    st.info('Cần thêm mo_hinh_demo.joblib từ Colab vào cùng thư mục app.py. Xem README.md.')
    st.stop()
try:
    bundle = load_bundle(path.stat().st_mtime_ns)
    meta = bundle.get('metadata', {})
    if (ROOT / 'metadata.json').exists():
        meta = json.loads((ROOT / 'metadata.json').read_text(encoding='utf-8'))
except Exception as exc:
    st.error('Không đọc được mô hình. Kiểm tra phiên bản thư viện theo requirements.txt xuất từ Colab.')
    st.code(str(exc)); st.stop()

st.session_state.setdefault('history', [])
st.title('Giám sát trạng thái máy')
st.caption('Phân tích thông số vận hành · Dự đoán bằng mô hình học máy')
a,b,c = st.columns(3)
a.metric('Mô hình', bundle.get('model_name', 'Chưa xác định'))
b.metric('Ngưỡng quyết định', f'{bundle["threshold"]:.2%}')
c.metric('Đầu vào', '6 thuộc tính')
if not meta.get('train_ranges'):
    st.warning('Chưa có phạm vi train. Chạy export_colab.py để bổ sung kiểm tra ngoài phạm vi; không dùng min/max từ toàn bộ dữ liệu thay thế.')

def remember(results, source):
    valid = results[results['prediction'].notna()].copy()
    valid['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    valid['source'] = source
    st.session_state['history'] = (st.session_state['history'] + valid.to_dict('records'))[-1000:]

def download(df, name, label='Tải kết quả CSV'):
    st.download_button(label, df.to_csv(index=False).encode('utf-8-sig'), name, 'text/csv')

def run_prediction(x, allow, source):
    try:
        result = predict_batch(bundle, x, meta, allow)
        remember(result, source)
        return result
    except Exception as exc:
        st.error('Dự đoán thất bại. Kiểm tra dữ liệu hoặc cấu hình mô hình.')
        st.code(str(exc))
        return None

if page == 'Dự đoán':
    st.subheader('Dự đoán một mẫu')
    sample = st.selectbox('Mẫu minh họa từ validation', ['Tự nhập', 'Mẫu không hỏng', 'Mẫu có hỏng'])
    d = {'Tự nhập':('M',300.,310.,1500,40.,100), 'Mẫu không hỏng':('M',302.5,312.,1487,46.1,175), 'Mẫu có hỏng':('L',304.,312.9,1363,62.5,200)}[sample]
    if st.session_state.get('last_sample') != sample:
        st.session_state.pop('single_result', None)
        st.session_state['last_sample'] = sample
    st.caption('Hai mẫu có sẵn đã được chọn trong các mẫu dự đoán đúng, chỉ để minh họa thao tác.')
    with st.form('input_' + sample):
        left,right=st.columns(2)
        typ=left.selectbox('Loại sản phẩm', ['L','M','H'], index=['L','M','H'].index(d[0]))
        air=left.number_input(LABELS[NUM[0]], value=d[1], step=.1)
        proc=left.number_input(LABELS[NUM[1]], value=d[2], step=.1)
        speed=right.number_input(LABELS[NUM[2]], value=d[3], step=1)
        torque=right.number_input(LABELS[NUM[3]], value=d[4], step=.1)
        wear=right.number_input(LABELS[NUM[4]], value=d[5], step=1)
        allow=st.checkbox('Cho phép thử ngoài phạm vi train (kết quả có thể thiếu tin cậy)')
        submitted=st.form_submit_button('Phân tích trạng thái', type='primary')
    if submitted:
        x=pd.DataFrame([[typ,air,proc,speed,torque,wear]],columns=FEATURES)
        result=run_prediction(x,allow,'single')
        if result is not None:
            st.session_state['single_result']=(result,x)
    if 'single_result' in st.session_state:
        result,x=st.session_state['single_result']; row=result.iloc[0]
        st.caption('Kết quả lần bấm gần nhất; sửa thông số cần bấm Phân tích lại.')
        if row['input_error']:
            st.error(row['input_error'])
        if row['range_warning']:
            st.warning(row['range_warning'])
        if pd.notna(row['prediction']):
            if row['prediction']==1: st.error('CÓ HỎNG HÓC — nhãn dự đoán 1')
            else: st.success('KHÔNG HỎNG HÓC — nhãn dự đoán 0')
            st.metric('Xác suất lớp hỏng do mô hình ước lượng',f'{row["probability"]:.2%}')
            st.caption('Không phải xác nhận an toàn hoặc xác suất rủi ro đã được kiểm chứng trên máy thật. Không dự đoán thời điểm hỏng.')
            st.caption('Đã hiệu chỉnh bằng sigmoid CV trên train.' if meta.get('calibrated') else 'Mô hình hiện chưa được hiệu chỉnh xác suất.')
            with st.expander('Vì sao mô hình đưa ra kết quả này?'):
                try:
                    contrib=contributions(bundle,x)
                    if contrib is None: st.info('Giải thích TreeSHAP chỉ khả dụng cho mô hình nền LightGBM trong gói này.')
                    else:
                        st.bar_chart(contrib.set_index('Đặc trưng'))
                        st.caption('TreeSHAP trên log-odds: dương đẩy về lớp hỏng, âm đẩy về lớp không hỏng. Không chứng minh nguyên nhân vật lý. Nếu có hiệu chỉnh, đây là giải thích mô hình nền, không phải phân rã xác suất sau hiệu chỉnh.')
                except Exception as exc: st.warning('Chưa tính được giải thích: '+str(exc))
            download(result,'du_doan_mot_mau.csv')
        elif not row['input_error']:
            st.info('Chưa dự đoán do ngoài phạm vi. Kiểm tra đầu vào hoặc xác nhận cho phép thử ngoài phạm vi.')
    with st.expander('Phạm vi từng thuộc tính trên train'):
        st.dataframe(pd.DataFrame(meta.get('train_ranges',{})).T)
        st.caption('Nằm trong từng khoảng không bảo đảm tổ hợp thông số có ý nghĩa vật lý hoặc dự đoán đúng.')
elif page=='CSV hàng loạt':
    st.subheader('Dự đoán nhiều mẫu')
    st.write('CSV cần sáu cột như mẫu. Cột nhãn hoặc mã định danh nếu có sẽ không được dùng làm đầu vào.')
    download(pd.DataFrame([['M',302.5,312.,1487,46.1,175]],columns=FEATURES),'mau_nhap.csv','Tải CSV mẫu')
    upload=st.file_uploader('Chọn CSV (tối đa 10 MB, 10.000 dòng)',type='csv')
    allow=st.checkbox('Cho phép dự đoán dòng ngoài phạm vi train')
    if st.button('Xử lý CSV', type='primary'):
        if upload is None: st.warning('Chọn tệp trước.')
        elif upload.size>10*1024*1024: st.error('Tệp vượt 10 MB.')
        else:
            try:
                frame=pd.read_csv(upload); frame.columns=frame.columns.str.strip()
                if not len(frame) or len(frame)>10000: raise ValueError('Cần từ 1 đến 10.000 dòng.')
                if frame.columns.duplicated().any(): raise ValueError('Tên cột bị trùng.')
                result=run_prediction(frame,allow,'csv')
                if result is not None: st.session_state['batch_result']=result
            except Exception as exc: st.error('Không đọc được CSV: '+str(exc))
    if 'batch_result' in st.session_state:
        r=st.session_state['batch_result']; st.dataframe(r,width='stretch')
        st.caption(f'Đã dự đoán {r.prediction.notna().sum()}/{len(r)} dòng. Dòng lỗi hoặc ngoài phạm vi bị chặn vẫn được giữ trong kết quả.')
        download(r,'du_doan_hang_loat.csv')
elif page=='Lịch sử':
    st.subheader('Lịch sử phiên hiện tại')
    st.caption('Tối đa 1.000 kết quả gần nhất. Không lưu bền vững; phiên mới hoặc khởi động lại có thể xóa lịch sử.')
    if st.button('Xóa lịch sử'): st.session_state['history']=[]
    hist=pd.DataFrame(st.session_state['history'])
    if hist.empty: st.info('Chưa có dự đoán.')
    else: st.dataframe(hist,width='stretch'); download(hist,'lich_su.csv')
elif page=='Đánh giá':
    st.subheader('Kết quả thực nghiệm')
    st.caption('Đọc kết quả đã xuất từ Colab; ứng dụng không huấn luyện hoặc điều chỉnh bằng tập test.')
    for filename,title in [('so_sanh_4_thuat_toan_validation.csv','So sánh cấu hình ban đầu'),('validation_comparison.csv','So sánh thực nghiệm mới'),('metrics.csv','Train / validation / test'),('ket_qua_test.csv','Kết quả test đã xuất'),('calibration_comparison.csv','Đánh giá hiệu chỉnh trên validation'),('cv_results.csv','Tìm kiếm siêu tham số trên train')]:
        f=ROOT/'reports'/filename
        if f.exists():
            st.markdown('**'+title+'**'); st.dataframe(pd.read_csv(f),width='stretch')
    for filename,title in [('pr_curve.png','Precision–Recall'),('overfit.png','So sánh train và validation'),('calibration.png','Độ phù hợp xác suất trên validation'),('confusion_test.png','Ma trận nhầm lẫn test'),('ma_tran_nham_lan_test.png','Ma trận nhầm lẫn test đã xuất')]:
        f=ROOT/'reports'/filename
        if f.exists(): st.image(str(f),caption=title)
    if not (ROOT/'reports').exists(): st.info('Chưa có báo cáo. Chạy export_colab.py hoặc train.py để xuất; không có số liệu giả thay thế.')
    st.info('AP và PR-AUC hình thang được báo cáo riêng. Khoảng cách train–validation là dấu hiệu cần phân tích, không tự chứng minh quá khớp. Test chỉ đánh giá sau khi khóa lựa chọn.')
else:
    st.subheader('Nguồn và giới hạn')
    st.write('AI4I 2020: dữ liệu tổng hợp mô phỏng bảo trì công nghiệp, UCI Machine Learning Repository.')
    st.link_button('Xem nguồn UCI','https://archive.ics.uci.edu/dataset/601/ai4i')
    st.write('Không kết nối máy thật. Không suy ra thời gian đến hỏng hoặc nguyên nhân vật lý từ dự đoán.')
    st.write('Chọn mô hình theo AP validation; chọn ngưỡng theo F1 validation. Hiệu chỉnh nếu có được học bằng CV trên train và chọn theo Brier score validation.')
    st.json({k:v for k,v in meta.items() if k!='train_ranges'})
