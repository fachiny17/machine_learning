import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump


def evaluate_regression(y_true, y_preds):
    """"Calculates and returns regression metrics"""
    metrics = {
        "R2": r2_score(y_true, y_preds),
        "MAE": mean_absolute_error(y_true, y_preds),
        "MSE": mean_squared_error(y_true, y_preds)
    }
    print(f"R2: {metrics['R2']:.2f}")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"MSE: {metrics['MSE']:.2f}")

    return metrics


# Load and prepare data-------------------------------------------------------------------

def train_and_save_model():
    # Setup random seed-------------------------------------------------------------------------
    np.random.seed(42)

    # Load data---------------------------------------------------------------------------------
    data = pd.read_csv("data/car-sales-extended-missing-data.csv")
    data.dropna(subset=["Price"], inplace=True)

    # Define features and transformerscategorical_features---------------------------------------
    categorical_features = ["Make", "Colour"]
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])

    door_feature = ["Doors"]
    door_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=4))])

    numeric_features = ["Odometer (KM)"]
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))])

    # Create preprocessing pipeline----------------------------------------------------------------
    preprocessor = ColumnTransformer(transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("door", door_transformer, door_feature),
        ("num", numeric_transformer, numeric_features)
    ])

    # Create full pipeline---------------------------------------------------------------------------
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100))
    ])

    # Split data-------------------------------------------------------------------------------------------
    x = data.drop("Price", axis=1)
    y = data["Price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Baseline model---------------------------------------------------------------------------------------
    print("Training baseline model...")
    baseline_model = model.fit(x_train, y_train)
    baseline_preds = baseline_model.predict(x_test)
    baseline_metrics = evaluate_regression(y_test, baseline_preds)

    # Hyperparameter grid----------------------------------------------
    param_grid = {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2]
    }

    # GridSearchCV------------------------------------------------------------------------------------
    print("\nStarting GridSearchCV...")
    gs_model = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        verbose=2,
        n_jobs=-1
    )
    gs_model.fit(x_train, y_train)

    # Best model-----------------------------------------------------------------------------
    print(f"\nBest parameters: {gs_model.best_params_}")
    best_model = gs_model.best_estimator_
    gs_preds = best_model.predict(x_test)
    gs_metrics = evaluate_regression(y_test, gs_preds)

    # Compare metrics --------------------------------------------------------------------
    compare_metrics = pd.DataFrame({
        "Baseline": baseline_metrics,
        "GridSearchCV": gs_metrics
    }).T
    print("\nModel Comparison:")
    print(compare_metrics)

    # Save model----------------------------------------------------------------------------------------------------
    dump(model, "models/car_price_model.joblib")
    print("Model trained and saved successfully!")


if __name__ == "__main__":
    train_and_save_model()
