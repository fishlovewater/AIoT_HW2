"""CRISP-DM: reproducible 50 Startups multiple linear regression.
Run locally: python train.py. Kaggle: paste this script into a notebook cell.
"""
import json
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

URL = 'https://raw.githubusercontent.com/harimittapalli/Mulitple-Linear-Reggression/master/50_Startups.csv'
ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def main():
    # 1. Business understanding: estimate profit from expenditure and location.
    # 2. Data understanding: fetch and validate the specified source.
    try:
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f'Unexpected HTTP status: {response.status_code}')
        data = pd.read_csv(StringIO(response.text))
        (ROOT / '50_Startups.csv').write_text(response.text, encoding='utf-8')
    except requests.RequestException as exc:
        if not (ROOT / '50_Startups.csv').exists():
            raise RuntimeError('Data retrieval failed; no local CSV is available.') from exc
        print('Network unavailable; using downloaded source CSV.')
        data = pd.read_csv(ROOT / '50_Startups.csv')
    numeric = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    if set(data.columns) != set(numeric + ['State']):
        raise ValueError(f'Unexpected columns: {list(data.columns)}')
    data[numeric] = data[numeric].apply(pd.to_numeric, errors='raise')
    if data.isna().any().any() or not np.isfinite(data[numeric]).all().all():
        raise ValueError('Dataset contains missing or non-finite values.')
    if len(data) != 50 or data.duplicated().any():
        raise ValueError('Expected 50 distinct startup observations.')
    print(data.head(), '\nColumns:', list(data.columns))
    # 3. Preparation: split before fitting preprocessing; California is reference.
    X, y = data.drop(columns='Profit'), data['Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    preparation = ColumnTransformer([('state', OneHotEncoder(drop='first', handle_unknown='error'), ['State'])], remainder='passthrough')
    # 4. Modeling: all three spend variables plus two state indicators.
    model = Pipeline([('prepare', preparation), ('regression', LinearRegression())])
    model.fit(X_train, y_train)
    # 5. Evaluation: held-out observations, no outlier deletion.
    predicted = model.predict(X_test)
    metrics = {'r2': float(r2_score(y_test, predicted)), 'mae': float(mean_absolute_error(y_test, predicted)), 'mse': float(mean_squared_error(y_test, predicted)), 'rmse': float(np.sqrt(mean_squared_error(y_test, predicted)))}
    print(json.dumps(metrics, indent=2))
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predicted, color='#197a59')
    bounds = [min(y_test.min(), predicted.min()), max(y_test.max(), predicted.max())]
    plt.plot(bounds, bounds, '--', color='#999999')
    plt.xlabel('Actual Profit'); plt.ylabel('Predicted Profit'); plt.title('Actual vs. Predicted Profit')
    plt.tight_layout(); plt.savefig(ROOT / 'actual-vs-predicted.png', dpi=180); plt.close()
    # 6. Deployment: export exact fitted coefficients for the browser calculator.
    reg = model.named_steps['regression']
    states = model.named_steps['prepare'].named_transformers_['state'].categories_[0].tolist()
    rows = []
    all_predictions = model.predict(X)
    for i, row in data.iterrows():
        rows.append({'id': int(i), 'rd': row['R&D Spend'], 'admin': row['Administration'], 'marketing': row['Marketing Spend'], 'state': row['State'], 'profit': row['Profit'], 'predicted': float(all_predictions[i]), 'split': 'test' if i in X_test.index else 'train'})
    result = {'source': URL, 'metrics': metrics, 'intercept': float(reg.intercept_), 'coefficients': reg.coef_.tolist(), 'states': states, 'rows': rows, 'correlation': data[numeric].corr().values.tolist(), 'testIds': X_test.index.tolist(), 'trainSize': len(X_train), 'testSize': len(X_test), 'seed': 0}
    (ROOT / 'model.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    main()
