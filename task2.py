import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)


customer_ids = list(range(1, 201))


genders = ['Male'] * 100 + ['Female'] * 100
np.random.shuffle(genders)


ages = np.random.randint(18, 71, 200)


incomes = np.random.randint(15, 141, 200)


spending_scores = []
for age, income in zip(ages, incomes):
    base_score = 100 - (age / 2) + (income / 3)
    base_score = max(1, min(100, base_score))
    spending_scores.append(int(base_score + np.random.normal(0, 10)))

spending_scores = [max(1, min(100, score)) for score in spending_scores]


data = pd.DataFrame({
    'CustomerID': customer_ids,
    'Gender': genders,
    'Age': ages,
    'Annual Income (k$)': incomes,
    'Spending Score (1-100)': spending_scores
})

print("Mall Customer Dataset Overview:")
print("=" * 50)
print(f"Dataset shape: {data.shape}")
print("\nFirst 10 rows:")
print(data.head(10))
print("\nDataset information:")
print(data.info())
print("\nDescriptive statistics:")
print(data.describe())
print("\n" + "=" * 50 + "\n")


print("Missing values check:")
print(data.isnull().sum())
print("\n" + "=" * 50 + "\n")


data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features after standardization:")
print(pd.DataFrame(X_scaled, columns=features).head())
print("\n" + "=" * 50 + "\n")

wcss = []  
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    
    if k > 1:  
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, 'ro-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k values')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


optimal_k = 5  


kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)


data['Cluster'] = kmeans.labels_

print(f"K-means clustering completed with {optimal_k} clusters")
print("\nCluster distribution:")
print(data['Cluster'].value_counts().sort_index())
print("\n" + "=" * 50 + "\n")


cluster_summary = data.groupby('Cluster')[features + ['Gender']].mean()
cluster_summary['Count'] = data['Cluster'].value_counts()

print("Cluster Summary:")
print(cluster_summary)
print("\n" + "=" * 50 + "\n")


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


cluster_viz = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
cluster_viz['Cluster'] = data['Cluster']

plt.figure(figsize=(15, 12))


plt.subplot(2, 2, 1)
scatter = plt.scatter(cluster_viz['PC1'], cluster_viz['PC2'], c=cluster_viz['Cluster'], 
                     cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Customer Clusters (PCA Visualization)')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 2)
scatter = plt.scatter(data['Age'], data['Spending Score (1-100)'], 
                     c=data['Cluster'], cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Age vs Spending Score by Cluster')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 3)
scatter = plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], 
                     c=data['Cluster'], cmap='viridis', alpha=0.7, s=50)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Income vs Spending Score by Cluster')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 4)
cluster_counts = data['Cluster'].value_counts()
plt.bar(cluster_counts.index, cluster_counts.values, color=plt.cm.viridis(np.linspace(0, 1, optimal_k)))
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.title('Customer Distribution Across Clusters')
plt.xticks(range(optimal_k))

plt.tight_layout()
plt.show()


print("Detailed Cluster Analysis:")
print("=" * 40)

cluster_descriptions = {
    0: "Young customers with moderate income and high spending",
    1: "Middle-aged customers with high income and high spending",
    2: "Older customers with moderate income and low spending",
    3: "Young customers with low income and low spending",
    4: "Middle-aged customers with moderate income and moderate spending"
}

for cluster in range(optimal_k):
    cluster_data = data[data['Cluster'] == cluster]
    print(f"\nCluster {cluster}: {cluster_descriptions.get(cluster, 'Unknown')}")
    print(f"Number of customers: {len(cluster_data)}")
    print(f"Average Age: {cluster_data['Age'].mean():.1f} years")
    print(f"Average Annual Income: ${cluster_data['Annual Income (k$)'].mean():.1f}k")
    print(f"Average Spending Score: {cluster_data['Spending Score (1-100)'].mean():.1f}")
    gender_dist = cluster_data['Gender'].value_counts()
    male_count = gender_dist.get(0, 0)
    female_count = gender_dist.get(1, 0)
    print(f"Gender distribution: {male_count} Male, {female_count} Female")


print("\n" + "=" * 50)
print("CUSTOMER SEGMENTATION INSIGHTS:")
print("=" * 50)
print("1. High-Value Customers: Clusters with high spending scores")
print("2. Budget-Conscious: Clusters with low spending scores")
print("3. Target for Upselling: Moderate spenders who could spend more")
print("4. Demographic Patterns: Age and income strongly influence spending behavior")
print("5. Marketing Strategy: Different clusters require tailored marketing approaches")


data.to_csv('customer_segmentation_results.csv', index=False)
print("\nResults saved to 'customer_segmentation_results.csv'")


new_customers = pd.DataFrame({
    'Age': [25, 45, 60],
    'Annual Income (k$)': [40, 80, 30],
    'Spending Score (1-100)': [70, 20, 50]
})

new_customers_scaled = scaler.transform(new_customers)
predicted_clusters = kmeans.predict(new_customers_scaled)

print("\nNew Customer Predictions:")
for i, (age, income, score, cluster) in enumerate(zip(new_customers['Age'], 
                                                     new_customers['Annual Income (k$)'], 
                                                     new_customers['Spending Score (1-100)'], 
                                                     predicted_clusters)):
    print(f"Customer {i+1}: Age={age}, Income=${income}k, Score={score} → Cluster {cluster} ({cluster_descriptions[cluster]})")


print("\n" + "=" * 50)
print("SAMPLE CUSTOMERS FROM EACH CLUSTER:")
print("=" * 50)

for cluster in range(optimal_k):
    cluster_samples = data[data['Cluster'] == cluster].head(3)
    print(f"\nCluster {cluster} Samples:")
    for _, row in cluster_samples.iterrows():
        gender = "Male" if row['Gender'] == 0 else "Female"
        print(f"  Customer {int(row['CustomerID'])}: {gender}, {int(row['Age'])} yrs, "
              f"${int(row['Annual Income (k$)'])}k, Score: {int(row['Spending Score (1-100)'])}")
