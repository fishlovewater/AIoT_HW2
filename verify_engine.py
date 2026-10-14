import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from pathlib import Path
root=Path(__file__).parent
result=json.loads((root/'engine-check.json').read_text())
data=pd.read_csv(root/'50_Startups.csv')
test_ids=[p['id']-2 for p in result['models'][0]['predictions']]
train_ids=[i for i in range(len(data)) if i not in test_ids]
X=pd.get_dummies(data.drop(columns='Profit'),drop_first=True).astype(float)
scale=StandardScaler().fit(X.iloc[train_ids])
Xt=scale.transform(X.iloc[train_ids]); Xv=scale.transform(X.iloc[test_ids])
y=data.Profit.iloc[train_ids].to_numpy(); ym=y.mean(); ys=y.std(); yz=(y-ym)/ys
models=[LinearRegression(),Ridge(alpha=len(train_ids)*.1),Lasso(alpha=.1,tol=1e-10,max_iter=10000)]
for expected,model in zip(result['models'],models):
    model.fit(Xt,yz)
    pred=model.predict(Xv)*ys+ym
    browser=np.array([p['predicted'] for p in expected['predictions']])
    assert np.max(np.abs(pred-browser))<.01, (expected['type'],np.max(np.abs(pred-browser)))
    rmse=np.sqrt(mean_squared_error(data.Profit.iloc[test_ids],pred))
    assert abs(rmse-expected['rmse'])<.01
    print(expected['type'], 'PASS: sklearn predictions and RMSE within 0.01')
