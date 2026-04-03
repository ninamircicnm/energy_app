# data_processing.py
import pandas as pd

# Load data and take a sample
def load_data(file_path, frac=0.1):
    data = pd.read_csv(file_path)
    data_sample = data.sample(frac=frac, random_state=101)
    return data_sample

# Display a preview of the sampled data
def show_head(data_sample, n=10):
    return data_sample.head(n)
