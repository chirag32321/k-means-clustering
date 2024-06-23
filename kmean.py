#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd


# In[3]:


data=pd.read_csv("Mall_Customers.csv")
data


# In[7]:


purchase = data[['Spending Score (1-100)']]


# In[8]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(purchase)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.show()


# In[15]:


k = 2

kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(purchase)


# In[20]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

cluster_data = data[['Spending Score (1-100)', 'Annual Income (k$)']]



scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(n_clusters=2, random_state=42)
cluster_data['Cluster'] = kmeans.fit_predict(cluster_data_scaled)

cluster_data['Cluster Name'] = cluster_data['Cluster']
    
plt.figure(figsize=(10, 6))
colors = [ 'pink', 'blue']
for i, cluster in enumerate(cluster_data['Cluster'].unique()):
    subset = cluster_data[cluster_data['Cluster'] == cluster]
    plt.scatter(subset['Spending Score (1-100)'], subset['Annual Income (k$)'], label=f'Cluster {cluster}: {subset["Cluster Name"].iloc[0]}', color=colors[i])

plt.xlabel('Purchase History')
plt.ylabel('Annual Income (k$)')
plt.legend()
plt.title('Purchase History vs. Annual Income')
plt.show()


# In[ ]:




