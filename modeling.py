# modeling.py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#Return the subset of data belonging to the largest cluster
def get_largest_cluster(df):
    largest_cluster = df['Cluster'].value_counts().idxmax()
    return df[df['Cluster'] == largest_cluster]

#Split the DataFrame into features (X) and target (y)
def prepare_regression_data(df, target_column):
    X = df.drop(columns=[target_column, 'Cluster'])
    X = X.select_dtypes(include=["int64", "float64"])
    
    y = df[target_column]
    return X, y

#Scale features and train an SVR model, returning predictions
def train_svr_model(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=101
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(kernel='rbf')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    return model, scaler, y_test, y_pred

#Print and return MAE, RMSE, and R² metrics
def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("MAE:", mae)    # Mean absolute prediction error
    print("RMSE:", rmse)  # Penalises large errors (outliers)
    print("R²:", r2)      # Coefficient of determination

    return mae, rmse, r2

#Scatter plot of actual vs. predicted values
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--')
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("SVR – Actual vs. Predicted Values")
    plt.grid(True)

#Run SVR regression on the largest cluster
#Note: with large clusters this can take a few minutes
def run_svm_regression(df, target_column="Total_Electricity_Energy"):
    if "Cluster" not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column.")

    if target_column not in df.columns:
        raise ValueError(f"Target column missing: {target_column}")

    df_largest = get_largest_cluster(df)
    X, y = prepare_regression_data(df_largest, target_column)
    model, scaler, y_test, y_pred = train_svr_model(X, y)
    mae, rmse, r2 = evaluate_regression(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Sample size": len(df_largest),
        "y_test": y_test,
        "y_pred": y_pred
    }
