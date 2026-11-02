"""Train-only CV/tuning/calibration; validation selects; locked test evaluation."""
import argparse,json,sys,hashlib
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from sklearn.metrics import average_precision_score,precision_recall_curve,brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from core import FEATURES,NUM,validate
from evaluation import measure,build_reports

def prep(scale):
    return ColumnTransformer([('numeric',StandardScaler() if scale else 'passthrough',NUM),('type',OneHotEncoder(categories=[['L','M','H']],drop=['L'],sparse_output=False,handle_unknown='error'),['Type'])])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--csv',required=True);parser.add_argument('--output',default='trained_run');parser.add_argument('--advanced',action='store_true');args=parser.parse_args()
    dest=Path(args.output)
    if (dest/'mo_hinh_demo.joblib').exists():raise SystemExit('Thư mục đã có mô hình. Chọn --output mới để không ghi đè thực nghiệm cũ.')
    dest.mkdir(parents=True,exist_ok=True);reports=dest/'reports';reports.mkdir(exist_ok=True)
    raw=Path(args.csv).read_bytes();df=pd.read_csv(args.csv);df.columns=df.columns.str.strip()
    x,errors,_=validate(df,{})
    if errors.ne('').any():raise ValueError('CSV có lỗi đầu vào: '+str(errors[errors.ne('')].head().to_dict()))
    y=pd.to_numeric(df['Machine failure'],errors='raise')
    if y.isna().any() or not y.isin([0,1]).all():raise ValueError('Nhãn phải là 0/1, không thiếu.')
    y=y.astype(int)
    xt,xremain,yt,yremain=train_test_split(x,y,test_size=.3,stratify=y,random_state=42)
    xv,xs,yv,ys=train_test_split(xremain,yremain,test_size=.5,stratify=yremain,random_state=42)
    ratio=float((yt==0).sum()/(yt==1).sum()); models={}
    for weighted in [False,True]:
        suffix=' - weighted' if weighted else ''
        models['Logistic Regression'+suffix]=Pipeline([('preprocessing',prep(True)),('model',LogisticRegression(max_iter=2000,class_weight='balanced' if weighted else None,random_state=42))])
        models['Random Forest'+suffix]=Pipeline([('preprocessing',prep(False)),('model',RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced' if weighted else None,random_state=42,n_jobs=2))])
        models['XGBoost'+suffix]=Pipeline([('preprocessing',prep(False)),('model',XGBClassifier(n_estimators=300,learning_rate=.05,max_depth=4,min_child_weight=1,subsample=.8,colsample_bytree=.8,reg_lambda=1,scale_pos_weight=ratio if weighted else 1,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=42,n_jobs=2))])
        models['LightGBM'+suffix]=Pipeline([('preprocessing',prep(False)),('model',LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=15,max_depth=4,min_child_samples=20,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=1,scale_pos_weight=ratio if weighted else 1,random_state=42,n_jobs=2,verbosity=-1))])
    records=[]
    for name,model in models.items():
        print('Training:',name,flush=True);model.fit(xt,yt)
        p=model.predict_proba(xv)[:,1]
        records.append({'Model':name,**measure(yv,p,.5)})
    cv=StratifiedKFold(3,shuffle=True,random_state=42)
    if args.advanced:
        # Tuning scope explicitly limited to LightGBM; other models remain baselines.
        search=RandomizedSearchCV(models['LightGBM'],{
            'model__n_estimators':[200,400,600], 'model__learning_rate':[.03,.05,.1],
            'model__max_depth':[3,4,6], 'model__num_leaves':[7,15,31],
            'model__min_child_samples':[10,20,40], 'model__reg_lambda':[1.,5.,10.],
            'model__scale_pos_weight':[1.,ratio**.5,ratio]},n_iter=12,scoring='average_precision',cv=cv,random_state=42,n_jobs=1,refit=True)
        print('Train-only CV: 12 candidates x 3 folds',flush=True);search.fit(xt,yt)
        pd.DataFrame(search.cv_results_).to_csv(reports/'cv_results.csv',index=False)
        (reports/'best_params.json').write_text(json.dumps(search.best_params_,indent=2),encoding='utf-8')
        models['LightGBM tuned']=search.best_estimator_
        p=search.best_estimator_.predict_proba(xv)[:,1]
        records.append({'Model':'LightGBM tuned',**measure(yv,p,.5)})
    comparison=pd.DataFrame(records).sort_values('AP',ascending=False,kind='stable');comparison.to_csv(reports/'validation_comparison.csv',index=False)
    name=comparison.iloc[0]['Model'];base=models[name];chosen=base;calibrated=False
    if args.advanced:
        # Each calibration fold refits the complete pipeline: no global scaler leakage.
        cal=CalibratedClassifierCV(base,method='sigmoid',cv=cv,ensemble=True,n_jobs=1);cal.fit(xt,yt)
        p0=base.predict_proba(xv)[:,1];p1=cal.predict_proba(xv)[:,1]
        cal_rows=[{'Variant':'Original','Brier':brier_score_loss(yv,p0),'AP':average_precision_score(yv,p0)},{'Variant':'Sigmoid CV','Brier':brier_score_loss(yv,p1),'AP':average_precision_score(yv,p1)}]
        pd.DataFrame(cal_rows).to_csv(reports/'calibration_comparison.csv',index=False)
        fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'--',color='gray')
        for label,p in [('Original',p0),('Sigmoid CV',p1)]:
            actual,pred=calibration_curve(yv,p,n_bins=6,strategy='quantile');ax.plot(pred,actual,'o-',label=label)
        ax.set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Calibration on validation');ax.legend();fig.tight_layout();fig.savefig(reports/'calibration.png',dpi=160);plt.close(fig)
        if cal_rows[1]['Brier']<cal_rows[0]['Brier']:chosen=cal;calibrated=True
    p=chosen.predict_proba(xv)[:,1];pr,re,thresholds=precision_recall_curve(yv,p);f1=2*pr[:-1]*re[:-1]/np.maximum(pr[:-1]+re[:-1],1e-12);idx=np.flatnonzero(f1==f1.max())[-1];threshold=float(thresholds[idx])
    metadata={'dataset':'AI4I 2020 synthetic','dataset_sha256':hashlib.sha256(raw).hexdigest(),'split_seed':42,'train_ranges':{col:{'min':float(xt[col].min()),'max':float(xt[col].max())} for col in NUM},'calibrated':calibrated,'selection':'AP validation; calibration by Brier validation; threshold by F1 validation','advanced':args.advanced,'python':sys.version,'versions':{p:version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','xgboost','joblib']}}
    bundle={'pipeline':chosen,'explanation_pipeline':base,'threshold':threshold,'model_name':name+(' + sigmoid CV' if calibrated else ''),'feature_columns':FEATURES,'metadata':metadata}
    # Freeze selection before accessing test probabilities.
    joblib.dump(bundle,dest/'mo_hinh_demo.joblib');(dest/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    table=build_reports(bundle,{'Train':(xt,yt),'Validation':(xv,yv),'Test':(xs,ys)},reports)
    print(table.to_string(index=False))
    with open(dest/'requirements.txt','w') as f:
        f.write('streamlit>=1.40,<2\nmatplotlib>=3.8,<4\n')
        for pkg,ver in metadata['versions'].items():f.write(f'{pkg}=={ver}\n')
    for split,(xx,yy) in {'train':(xt,yt),'validation':(xv,yv),'test':(xs,ys)}.items():
        pd.DataFrame({'source_row':xx.index,'label':np.asarray(yy)}).to_csv(reports/f'{split}_indices.csv',index=False)
    print('Saved',dest)
if __name__=='__main__': main()
