import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

root = os.path.dirname(__file__)
source_path = os.path.join(root, 'data', 'picking_time_data.csv')
output_dir = os.path.join(root, 'Outputs')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(source_path)
X = df.drop(columns=['picking_time'])
y = df['picking_time']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = [
    ('Linear Regression', LinearRegression(), X_train, X_test),
    ('Ridge Regression', Ridge(random_state=42), X_train_scaled, X_test_scaled),
    ('Random Forest Regressor', RandomForestRegressor(random_state=42), X_train, X_test),
    ('MLP Regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42), X_train_scaled, X_test_scaled),
]

results = []
trained_models = {}
for name, model, x_train, x_test in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results.append({
        'model': name,
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
    })
    trained_models[name] = model

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(output_dir, 'Task4_model_comparison.csv'), index=False)

best_row = results_df.loc[results_df['rmse'].idxmin()]
best_name = best_row['model']
best_model = trained_models[best_name]

print('Dataset shape:', df.shape)
print('X_train shape:', X_train.shape, 'X_test shape:', X_test.shape)
print(results_df.to_string(index=False))
print('Best model:', best_name)

if hasattr(trained_models['MLP Regressor'], 'loss_curve_'):
    plt.figure()
    plt.plot(trained_models['MLP Regressor'].loss_curve_)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Training Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Task2_training_curve.png'))
    plt.close()

best_test = X_test_scaled if best_name in ('Ridge Regression', 'MLP Regressor') else X_test
best_pred = best_model.predict(best_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=1)
plt.xlabel('Actual picking_time')
plt.ylabel('Predicted picking_time')
plt.title(f'Prediction vs Actual - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Task3_prediction_vs_actual.png'))
plt.close()

residuals = y_test - best_pred
plt.figure(figsize=(6, 4))
plt.scatter(best_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linewidth=1)
plt.xlabel('Predicted picking_time')
plt.ylabel('Residuals')
plt.title(f'Residual Plot - {best_name}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Task6_residual_plot.png'))
plt.close()

plt.figure(figsize=(6, 4))
rf = trained_models['Random Forest Regressor']
importances = rf.feature_importances_
features = X.columns.tolist()
indices = importances.argsort()[::-1]
plt.bar([features[i] for i in indices], importances[indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Task7_feature_importance.png'))
plt.close()

plt.figure(figsize=(6, 4))
plt.bar(results_df['model'], results_df['rmse'], color=['#4c72b0', '#55a868', '#c44e52', '#8172b2'])
plt.ylabel('RMSE')
plt.title('Model Comparison by RMSE')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Task5_model_comparison.png'))
plt.close()
