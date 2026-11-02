"""Run inside the existing notebook with exec(open(...).read())."""
from pathlib import Path
from importlib.metadata import version
import json,sys,shutil,joblib
from core import NUM
from evaluation import build_reports
required=['pipeline_final','nguong_tot_nhat','ten_tot_nhat','X_train','y_train','X_val','y_val','X_test','y_test']
missing=[name for name in required if name not in globals()]
if missing:raise RuntimeError('Thiếu biến notebook: '+', '.join(missing)+'. Chạy các ô gốc trước.')
package=Path(__file__).parent if '__file__' in globals() else Path('nhom16_upgrade')
# exec uses notebook __file__ inconsistently: locate files by their known directory.
if not (package/'app.py').exists():package=Path('nhom16_upgrade')
meta={'dataset':'AI4I 2020 synthetic','split_seed':42,'calibrated':False,'train_ranges':{col:{'min':float(X_train[col].min()),'max':float(X_train[col].max())} for col in NUM},'python':sys.version,'selection':'Existing frozen model: validation AP; threshold F1 validation','versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','xgboost','joblib']}}
pack={'pipeline':pipeline_final,'threshold':float(nguong_tot_nhat),'model_name':ten_tot_nhat,'feature_columns':list(X_train.columns),'metadata':meta}
joblib.dump(pack,package/'mo_hinh_demo.joblib');(package/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
r=package/'reports';r.mkdir(exist_ok=True)
# Fixed model only: recomputes descriptive plots, never changes model or threshold.
build_reports(pack,{'Train':(X_train,y_train),'Validation':(X_val,y_val),'Test':(X_test,y_test)},r)
if 'bang_so_sanh' in globals():bang_so_sanh.to_csv(r/'so_sanh_4_thuat_toan_validation.csv',index=False)
with open(package/'requirements.txt','w') as f:
    f.write('streamlit>=1.40,<2\nmatplotlib>=3.8,<4\n')
    for p,v in meta['versions'].items():f.write(f'{p}=={v}\n')
archive=shutil.make_archive('Nhom16_Web_Upgrade','zip',root_dir=str(package))
print('Đã xuất mô hình giữ nguyên, phạm vi train, biểu đồ và mã web:',archive)
from google.colab import files
files.download(archive)
