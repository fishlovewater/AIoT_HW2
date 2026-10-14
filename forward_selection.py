"""Reproduce the requested forward selection; comparison split is reused for selection."""
import json
from io import StringIO
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

URL = 'https://gist.githubusercontent.com/GaneshSparkz/b5662effbdae8746f7f7d8ed70c42b2d/raw/faf8b1a0d58e251f48a647d3881e7a960c3f0925/50_Startups.csv'
ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def evaluate_performance(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred))), float(r2_score(y_true, y_pred))

def forward_selection(X_train, X_test, y_train, y_test, max_features=5):
    if not 1 <= max_features <= len(X_train.columns):
        raise ValueError('max_features must be between 1 and the available feature count')
    selected, results = [], []
    for count in range(1, max_features + 1):
        candidates = []
        for feature in X_train.columns.difference(selected):
            features = selected + [feature]
            model = LinearRegression().fit(X_train[features], y_train)
            predictions = model.predict(X_test[features])
            rmse, r2 = evaluate_performance(y_test, predictions)
            candidates.append({'feature': feature, 'rmse': rmse, 'r2': r2, 'model': model, 'predictions': predictions})
        best = min(candidates, key=lambda c: (c['rmse'], -c['r2']))
        selected.append(best['feature'])
        results.append({'count': count, 'features': selected.copy(), 'added': best['feature'], 'rmse': best['rmse'], 'r2': best['r2'], 'intercept': float(best['model'].intercept_), 'coefficients': best['model'].coef_.tolist(), 'predictions': [{'id': int(i), 'actual': float(actual), 'predicted': float(pred)} for i, actual, pred in zip(y_test.index, y_test, best['predictions'])], 'candidates': [{'feature': c['feature'], 'rmse': c['rmse'], 'r2': c['r2']} for c in candidates]})
    return results

def main():
    try:
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError(f'Unexpected status {response.status_code}')
        data = pd.read_csv(StringIO(response.text))
        (ROOT / 'selection-data.csv').write_text(response.text, encoding='utf-8')
    except requests.RequestException as exc:
        if not (ROOT / 'selection-data.csv').exists():
            raise RuntimeError('Source download failed and no cached data exists') from exc
        print('Network unavailable; using downloaded source CSV.')
        data = pd.read_csv(ROOT / 'selection-data.csv')
    numeric = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']
    if set(data.columns) != set(numeric + ['State']):
        raise ValueError('Unexpected dataset columns')
    data[numeric] = data[numeric].apply(pd.to_numeric, errors='raise')
    if len(data) != 50 or data.isna().any().any() or data.duplicated().any() or not np.isfinite(data[numeric]).all().all():
        raise ValueError('Expected 50 complete, unique, finite observations')
    X = pd.get_dummies(data.drop(columns='Profit'), columns=['State'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, data['Profit'], test_size=0.2, random_state=0)
    results = forward_selection(X_train, X_test, y_train, y_test)
    markdown = '| Number of Features | Selected Features | RMSE (USD) | R-squared |\n| --- | --- | ---: | ---: |\n'
    for row in results:
        markdown += f"| {row['count']} | {', '.join(row['features'])} | {row['rmse']:.6f} | {row['r2']:.6f} |\n"
    print(markdown)
    (ROOT / 'selection-results.md').write_text(markdown, encoding='utf-8')
    result = {'source': URL, 'trainSize': len(X_train), 'comparisonSize': len(X_test), 'seed': 0, 'results': results, 'bestCount': min(results, key=lambda r: r['rmse'])['count']}
    (ROOT / 'selection.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
