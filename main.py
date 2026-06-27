import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

housing = fetch_california_housing(as_frame=True)
df = housing.frame


df['Bedrooms_Ratio'] = df['AveBedrms'] / df['AveRooms']
df['People_Per_Room'] = df['Population'] / df['AveRooms']

# Координаты Лос-Анджелеса: 34.05, -118.24
df['Dist_to_LA'] = np.sqrt((df['Latitude'] - 34.05)**2 + (df['Longitude'] - (-118.24))**2)

# Координаты Сан-Франциско: 37.77, -122.41
df['Dist_to_SF'] = np.sqrt((df['Latitude'] - 37.77)**2 + (df['Longitude'] - (-112.41))**2)

X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Форма тренировочной выборки: ", X_train.shape)
print("Форма тестовой выборки: ", X_test.shape)


base_CatBoost = CatBoostRegressor(verbose=0, random_state=42)

param_grid = {
    'depth': [8],
    'iterations': [1000],
    'learning_rate': [0.1],
    'l2_leaf_reg': [3]
}

grid_search = GridSearchCV(
    estimator=base_CatBoost,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=0,
)

grid_search.fit(X_train, y_train)

best_CatBoost = grid_search.best_estimator_

print(f'лучшие параметры: {grid_search.best_params_}')
print(f'лучшая метрика: {grid_search.best_score_}')

rf_model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

pred_CatBoost = best_CatBoost.predict(X_test)
pred_rf = rf_model.predict(X_test)

final_prediction = (pred_CatBoost + pred_rf) / 2


mae = mean_absolute_error(y_test, final_prediction)
mae_in_dollars = mae * 100000

print("\n-- Результаты оценки ---")
print(f"Средняя абсолютная ошибка: {mae:.4f}")
print(f"Ошибка в долларах: ${mae_in_dollars:.2f}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, final_prediction, alpha=0.3, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', lw=2)
plt.title('Реальные сцены VS Предсказания модели')
plt.xlabel('Реальная стоимость жилья')
plt.ylabel('Предсказанная стоимость')
plt.grid(True)
plt.show()
