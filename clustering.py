# clustering.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

#Select relevant features and scale numeric variables
def prepare_features(df):
    selected_features = [
        "Total_Building_Area",
        "Total_Electricity_Energy",
        "Domestic_Hot_Water_Usage",
        "Floor_Insulation_U-Value",
        "Roof_Insulation_U-Value",
        "Window_Insulation_U-Value",
        "Door_Insulation_U-Value",
        "Energy_Use_Intensity",
        "Total_Heating_Energy"
    ]
    X = df[selected_features].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.dropna()

    if X.empty:
        raise ValueError("No numeric columns available for clustering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled

#Evaluate cluster quality using silhouette scores on a subsample
def silhouette_method(X_scaled, k_values=range(3, 7), sample_size=2000):
    scores = {}

    if X_scaled.shape[0] > sample_size:
        idx = np.random.choice(X_scaled.shape[0], sample_size, replace=False)
        X_used = X_scaled[idx]
    else:
        X_used = X_scaled

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=101)
        labels = kmeans.fit_predict(X_used)
        score = silhouette_score(X_used, labels)
        scores[k] = score

    return scores

#Visualize clusters for two selected features
def plot_clusters(df, x_feature="Energy_Use_Intensity",
                  y_feature="Total_Electricity_Energy"):
    if "Cluster" not in df.columns:
        raise ValueError("DataFrame must contain a 'Cluster' column.")

    if x_feature not in df.columns or y_feature not in df.columns:
        raise ValueError(
            f"Selected columns do not exist in the data: {x_feature}, {y_feature}"
        )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=x_feature,
        y=y_feature,
        hue="Cluster",
        palette="viridis",
        s=60
    )

    plt.title("Cluster Visualization (K-Means)")
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title="Cluster")
    plt.grid(True)

#Full KMeans pipeline: feature prep, silhouette scoring, clustering
def run_kmeans_pipeline(df, k_values=range(3, 7)):
    X_scaled = prepare_features(df)
    
    scores = silhouette_method(X_scaled, k_values)
    best_k = max(scores, key=scores.get)
    
    clustered_df = run_kmeans(df, X_scaled, best_k)

    return clustered_df, scores, best_k

#Train KMeans and append cluster labels to the original DataFrame
def run_kmeans(df, X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=101)
    labels = kmeans.fit_predict(X_scaled)

    df_with_clusters = df.copy()
    df_with_clusters["Cluster"] = labels

    return df_with_clusters
