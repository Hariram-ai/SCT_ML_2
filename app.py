import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)


# ==================================================
# Load Model and Data
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("kmeans_model.pkl")


@st.cache_data
def load_data():
    original_data = pd.read_csv("data/Mall_Customers.csv")
    clustered_data = pd.read_csv("customer_segments_result.csv")
    return original_data, clustered_data


model = load_model()
data, clustered_data = load_data()


# ==================================================
# Title
# ==================================================

st.title("🛍️ Mall Customer Segmentation")

st.write(
    "This application uses K-Means Clustering to segment "
    "customers based on Annual Income and Spending Score."
)

st.divider()


# ==================================================
# Customer Input
# ==================================================

st.header("Enter Customer Details")

income = st.slider(
    "Annual Income (k$)",
    min_value=int(data["Annual Income (k$)"].min()),
    max_value=int(data["Annual Income (k$)"].max()),
    value=50
)

spending_score = st.slider(
    "Spending Score (1-100)",
    min_value=int(data["Spending Score (1-100)"].min()),
    max_value=int(data["Spending Score (1-100)"].max()),
    value=50
)


# ==================================================
# Prediction
# ==================================================

if st.button("Predict Customer Segment"):

    # Create DataFrame with the same feature names
    # used when training the K-Means model
    input_data = pd.DataFrame(
        [[income, spending_score]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    # Predict cluster
    cluster = int(model.predict(input_data)[0])

    # Segment names
    segment_labels = {
    0: "Middle Income - Moderate Spending",
    1: "High Income - High Spending",
    2: "Young High Spenders",
    3: "High Income - Low Spending",
    4: "Low Income - Low Spending"
}

    segment_name = segment_labels.get(
        cluster,
        f"Cluster {cluster}"
    )

    # Display prediction
    st.success(
        f"Customer belongs to Cluster {cluster}: {segment_name}"
    )


    # ==================================================
    # Cluster Characteristics
    # ==================================================

    st.subheader("Cluster Characteristics")

    cluster_summary = clustered_data.groupby("Cluster")[
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ].mean()

    cluster_summary["Segment"] = cluster_summary.index.map(
        segment_labels
    )

    st.dataframe(
        cluster_summary.round(2),
        use_container_width=True
    )


    # ==================================================
    # Customer Segmentation Visualization
    # ==================================================

    st.subheader("Customer Segmentation")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Existing customers
    ax.scatter(
        clustered_data["Annual Income (k$)"],
        clustered_data["Spending Score (1-100)"],
        c=clustered_data["Cluster"],
        s=70,
        alpha=0.7
    )

    # Cluster centers
    ax.scatter(
        model.cluster_centers_[:, 0],
        model.cluster_centers_[:, 1],
        s=250,
        marker="X",
        edgecolor="black",
        label="Cluster Centers"
    )

    # Selected customer
    ax.scatter(
        income,
        spending_score,
        s=250,
        marker="*",
        edgecolor="black",
        label="Selected Customer"
    )

    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.set_title("Customer Segments")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ==================================================
# Dataset Preview
# ==================================================

st.divider()

st.subheader("Dataset Preview")

st.dataframe(
    data.head(10),
    use_container_width=True
)


# ==================================================
# Project Information
# ==================================================

st.divider()

st.subheader("About the Project")

st.write(
    """
    **Algorithm:** K-Means Clustering

    **Number of Clusters:** 5

    **Features Used:**
    - Annual Income (k$)
    - Spending Score (1-100)

    **Silhouette Score:** 0.5539
    """
)