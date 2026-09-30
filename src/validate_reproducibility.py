import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def train_and_predict():
    X_train=np.load("data/processed/X_train_final.npy"); X_test=np.load("data/processed/X_test_final.npy")
    y_train=np.load("data/processed/y_train.npy"); y_test=np.load("data/processed/y_test.npy")
    model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,class_weight="balanced")
    model.fit(X_train,y_train); pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
    return pred,prob,{"accuracy":accuracy_score(y_test,pred),"f1_score":f1_score(y_test,pred),"roc_auc":roc_auc_score(y_test,prob)}

def validate_reproducibility():
    a,pa,ma=train_and_predict(); b,pb,mb=train_and_predict()
    ok=np.array_equal(a,b) and np.allclose(pa,pb) and all(np.isclose(ma[k],mb[k]) for k in ma)
    print(f"Predictions identical : {np.array_equal(a,b)}")
    print(f"Probabilities identical: {np.allclose(pa,pb)}")
    print(f"Metrics identical     : {all(np.isclose(ma[k],mb[k]) for k in ma)}")
    if not ok: raise SystemExit(1)
    print(f"[SUCCESS] Reproducibility passed! Accuracy={ma['accuracy']:.4f} | F1={ma['f1_score']:.4f} | ROC-AUC={ma['roc_auc']:.4f}")

if __name__ == "__main__": validate_reproducibility()
