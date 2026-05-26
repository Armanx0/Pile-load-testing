import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

print("Loading data...")
# Read the NEW dataset
df = pd.read_csv('SPT Layered Soil.csv', encoding='ISO-8859-1')

# Set the target (notice it is different now)
TARGET = 'Pile bearing capacity Pu (kN)'

# Drop the 'S. No.' identifier column and the target to get our features
X = df.drop(columns=[TARGET, 'S. No.'])
y = df[TARGET]

# 3. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. SCALING (Crucial for the Web App)
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 5. XGBOOST & HYPERPARAMETER SPACE
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

param_grid = {
    "n_estimators": [50, 75, 100, 200, 300],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2, 0.3]
}

# 6. RANDOM SEARCH
print("Running Randomized Search (100 iterations x 10 folds = 1000 fits)...")
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=100,
    scoring="neg_mean_squared_error",
    cv=10,
    verbose=1, # Set to 1 to reduce console clutter, change to 2 for full details
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train_scaled, y_train)

best_model = random_search.best_estimator_
print(f"\nBest Parameters:\n{random_search.best_params_}")

# 7. EVALUATION
y_pred = best_model.predict(X_test_scaled)
print("\nModel Performance on Unseen Test Data:")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2   : {r2_score(y_test, y_pred):.4f}")

# 8. SAVE FILES
joblib.dump(scaler, "scaler.pkl")
joblib.dump(best_model, "xgboost_model.pkl")
print("\n✅ scaler.pkl and xgboost_model.pkl saved successfully!")