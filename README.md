# Mall Customer Segmentation using K-Means Clustering

## Project Overview

This project performs customer segmentation using the K-Means Clustering algorithm.

The objective is to group mall customers into meaningful segments based on their:

- Annual Income
- Spending Score

The project also includes an interactive Streamlit application that allows users to enter customer details and predict their customer segment.

## Dataset

The project uses the Mall Customers dataset containing 200 customer records.

### Dataset Features

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Customer spending score |

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Methodology

The project follows these steps:

1. Load the Mall Customers dataset.
2. Explore the dataset and check for missing values.
3. Select Annual Income and Spending Score as clustering features.
4. Use the Elbow Method to determine the appropriate number of clusters.
5. Apply K-Means Clustering.
6. Evaluate the clustering using the Silhouette Score.
7. Visualize the customer segments.
8. Analyze the characteristics of each cluster.
9. Save the trained K-Means model.
10. Develop an interactive Streamlit application.

## Choosing the Number of Clusters

The Elbow Method was used to determine the number of clusters.

The analysis indicated that **5 clusters** provide a suitable segmentation for this dataset.

## Model Evaluation

The Silhouette Score obtained for the K-Means model is:

**0.5539**

## Customer Segments

| Cluster | Segment | Average Income (k$) | Average Spending Score |
|---|---|---:|---:|
| 0 | Middle Income - Moderate Spending | 55.30 | 49.52 |
| 1 | High Income - High Spending | 86.54 | 82.13 |
| 2 | Young High Spenders | 25.73 | 79.36 |
| 3 | High Income - Low Spending | 88.20 | 17.11 |
| 4 | Low Income - Low Spending | 26.30 | 20.91 |

## Project Files

```text
Task_02_Customer_Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── screenshots/
│   ├── customer_segments.png
│   ├── elbow_method.png
│   └── silhouette_score.png
│
├── task2.ipynb
├── app.py
├── kmeans_model.pkl
├── customer_segments_result.csv
├── README.md
└── requirements.txt
## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Hariram-ai/SCT_ML_2.git

### 2. Navigate to the project directory

```bash
cd SCT_ML_2

### 3. Install dependencies

```bash
pip install -r requirements.txt

### 4. Run the Streamlit application

```bash
streamlit run app.py

## Live Demo

[Click here to access the live Streamlit application](https://sctml2-6ejd3zqnq8nxqckyq5xvnr.streamlit.app/)

## Results

The K-Means algorithm successfully divided the 200 customers into five clusters based on Annual Income and Spending Score.

The interactive Streamlit application allows users to enter Annual Income and Spending Score and obtain the corresponding customer segment.

## Author

**Hari Ram**