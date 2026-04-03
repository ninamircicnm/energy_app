# main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading

import pandas as pd
import matplotlib.pyplot as plt

from data_processing import load_data, show_head
import clustering as cl
from modeling import run_svm_regression, plot_predictions


class EnergyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Urban Building Energy Analysis")
        self.root.geometry("900x600")

        self.data = None
        self.clustered_data = None

        self.create_widgets()

    def create_widgets(self):
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        tk.Button(
            button_frame,
            text="Import Data",
            width=20,
            command=self.load_data_gui
        ).grid(row=0, column=0, padx=10)

        tk.Button(
            button_frame,
            text="K-Means Clustering",
            width=20,
            command=self.run_clustering_gui
        ).grid(row=0, column=1, padx=10)

        tk.Button(
            button_frame,
            text="SVM Regression",
            width=20,
            command=self.run_modeling_gui
        ).grid(row=0, column=2, padx=10)

        self.output = ScrolledText(self.root, width=110, height=30)
        self.output.pack(pady=10)

    def clear_output(self):
        self.output.delete(1.0, tk.END)

    def load_data_gui(self):
        self.clear_output()

        try:
            self.data = load_data("urban_building_stock_datasets.csv")
            head_table = show_head(self.data, 10)
            head_table = head_table.iloc[:, :5]

            self.output.insert(tk.END, "Data loaded successfully (10% sample)\n\n")
            self.output.insert(tk.END, "First 10 rows of the 10% sample:\n\n")
            self.output.insert(tk.END, head_table.to_string(index=False))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_clustering_gui(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please import data first!")
            return

        self.clear_output()
        self.output.insert(tk.END, "Running K-Means clustering...\n")
        self.output.insert(tk.END, "Please wait, this may take a moment...\n\n")
        self.root.update_idletasks()

        # Run in a separate thread to keep the UI responsive
        thread = threading.Thread(target=self.clustering_worker)
        thread.daemon = True
        thread.start()

    def clustering_worker(self):
        """Background worker for clustering."""
        try:
            self.clustered_data, scores, best_k = cl.run_kmeans_pipeline(self.data)
            self.root.after(0, self.display_clustering_results, scores, best_k)
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", str(e)))

    def display_clustering_results(self, scores, best_k):
        self.clear_output()
        plt.close('all')

        self.output.insert(tk.END, "Clustering complete!\n\n")
        self.output.insert(tk.END, "Silhouette scores:\n")
        for k, score in scores.items():
            self.output.insert(tk.END, f"K={k}: {score:.4f}\n")

        self.output.insert(tk.END, f"\nSelected K = {best_k}\n\n")
        self.output.insert(tk.END, "Number of instances per cluster:\n\n")
        self.output.insert(
            tk.END,
            self.clustered_data['Cluster'].value_counts().to_string()
        )

        cl.plot_clusters(self.clustered_data)
        plt.show()

    def run_modeling_gui(self):
        if self.clustered_data is None:
            messagebox.showwarning("Warning", "Please run clustering first!")
            return

        self.clear_output()
        plt.close('all')

        self.output.insert(tk.END, "Running SVM regression...\n")
        self.output.insert(tk.END, "Please wait, this may take a few minutes...\n")
        self.root.update_idletasks()

        thread = threading.Thread(target=self.modeling_worker)
        thread.daemon = True
        thread.start()

    def modeling_worker(self):
        """Background worker for SVR modeling."""
        try:
            results = run_svm_regression(self.clustered_data)
            self.root.after(0, lambda: self.display_modeling_results(results))
        except Exception as e:
            self.root.after(0, lambda e=e: messagebox.showerror("Error", str(e)))

    def display_modeling_results(self, results):
        self.clear_output()
        plt.close('all')

        self.output.insert(tk.END, "SVM Regression – Results\n\n")

        y_test = results.pop('y_test', None)
        y_pred = results.pop('y_pred', None)

        for key, value in results.items():
            self.output.insert(tk.END, f"{key}: {value}\n")

        if y_test is not None and y_pred is not None:
            plot_predictions(y_test, y_pred)
            plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = EnergyApp(root)
    root.mainloop()
