# Urban Building Energy Analysis

A desktop application for clustering and regression analysis of urban building energy data.

## Features
- Load and sample large CSV datasets
- K-Means clustering with silhouette-based optimal K selection
- SVR (Support Vector Regression) on the largest cluster
- Tkinter GUI with threaded background processing

## Dataset
Download the dataset from [Mendeley Data – Urban Building Energy Stock Datasets](https://data.mendeley.com/datasets/m6vv9k9gcd/1) and place the CSV file in the project root directory.

## Requirements
```
pip install pandas scikit-learn matplotlib seaborn
```

## Usage
```
python main.py
```

1. **Import Data** – loads 10% sample of the CSV dataset
2. **K-Means Clustering** – runs clustering and displays silhouette scores
3. **SVM Regression** – trains SVR on the largest cluster and shows evaluation metrics

## Project Structure
```
├── main.py            # GUI application entry point
├── data_processing.py # Data loading and sampling
├── clustering.py      # KMeans pipeline and visualisation
├── modeling.py        # SVR training, evaluation and plotting
```
