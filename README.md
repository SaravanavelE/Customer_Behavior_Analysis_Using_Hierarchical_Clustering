# Customer Behavior Analysis Using Hierarchical Clustering

## Project Overview
This project performs **Customer Behavior Analysis** using **Hierarchical Clustering** techniques on retail customer data. By analyzing **Annual Income** and **Spending Score**, the model groups customers into meaningful segments based on similarity, enabling better understanding of purchasing behavior.

Both **Agglomerative (bottom-up)** hierarchical clustering and **dendrogram visualization** are used to determine optimal customer clusters.

---

## Key Highlights
- Unsupervised learning using **Hierarchical Clustering**
- Dendrogram-based cluster interpretation
- Customer segmentation without predefined labels
- Visual analysis of customer spending behavior
- Suitable for retail and marketing analytics

---

## Dataset Source
- Dataset: `Mall_Customers.csv`

### Features Used
- `Annual_Income_(k$)`
- `Spending_Score`

*(Gender was label-encoded but not included in clustering features)*

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Google Colab

---

## Workflow
1. Load retail customer dataset
2. Encode categorical variables
3. Select relevant numerical features
4. Generate dendrogram using Ward’s method
5. Apply Agglomerative Hierarchical Clustering
6. Visualize customer clusters

---

## Model Training
```
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='average'
)
model.fit(d)
```
## Model Evaluation
Hierarchical clustering is an unsupervised algorithm, so evaluation is based on:
Dendrogram interpretation
Number of clusters formed
Visual separation of clusters
```
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(
    sch.linkage(d, method='ward')
)
```
## Performance Metrics
Number of Clusters: 5
Clustering Method: Agglomerative
Distance Metric: Euclidean
Linkage Strategy: Average / Ward (for dendrogram)

## Cluster Visualization
```
plt.figure(figsize=(15,7))
plt.scatter(
    d['Annual_Income_(k$)'],
    d['Spending_Score'],
    c=model.labels_
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.title('Hierarchical Clustering - Customer Segmentation')
plt.show()
```
## Project Structure
├── Mall_Customers.csv
├── customer_behavior_hierarchical_clustering.ipynb
├── README.md

## Future Enhancements
Apply feature scaling using StandardScaler
Compare Agglomerative vs Divisive clustering
Experiment with different linkage methods
Add silhouette score analysis
Combine hierarchical clustering with DBSCAN or K-Means
Extend analysis using more customer features

## Author
Saravanavel E
AI & Data Science Student
GitHub: https://github.com/SaravanavelE

## License
This project is intended for educational and academic purposes only.
