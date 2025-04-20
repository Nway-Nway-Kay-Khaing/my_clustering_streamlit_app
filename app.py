#app.py
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#Load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model  = pickle.load(f)
    
#Set the page config
st.set_page_config(page_title="k-Means Clustering App", layout="centered")

#Set title
st.title(" k-Means Clustering Visualizer By Nway Nway Kay Khaing")

#Display cluster centers
st.subheader(" Example Data for Visualization")
st.markdown("This demo uses example data(2D) to illustrate clustering results.")

#Load from a saved dataset ot generate synthetic data

X, _ = make_blobs(n_samples=300, centers=loaded_model.n_clusters, cluster_std=0.60, random_state=0)

#Predictusing the loaded model
y_kmeans=loaded_model.predict(X)



# Plotting
fig,ax = plt.subplots()
scatter=ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1], s=300, c='red', marker='.', label='Centroids')
ax.set_title('k-Means Clustering')
ax.legend()
st.pyplot(fig)
