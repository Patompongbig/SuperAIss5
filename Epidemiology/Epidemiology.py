import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('Train_set.csv')

# Calculate seasonality feature
df['Year_Week'] = df['Year'].astype(str) + '-' + df['Week_no'].astype(str)
weekly = df.groupby(['PROVINCE', 'Year', 'Week_no'])['Count'].mean().reset_index()
weekly_std = weekly.groupby(['PROVINCE', 'Week_no'])['Count'].std().reset_index(name='Weekly_Std')
weekly_avg = weekly.groupby(['PROVINCE', 'Week_no'])['Count'].mean().reset_index(name='Weekly_Avg')
seasonality_index = pd.merge(weekly_avg, weekly_std, on=['PROVINCE', 'Week_no'])
seasonality_index['Seasonality_Score'] = seasonality_index['Weekly_Std'] / seasonality_index['Weekly_Avg']
seasonality_feature = seasonality_index.groupby('PROVINCE')['Seasonality_Score'].mean().reset_index()

# Focus only on the seasonality feature
X = seasonality_feature[['Seasonality_Score']].values

# Normalize using RobustScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(seasonality_feature[['Seasonality_Score']].values)

# Define the SOM and train using only Seasonality_Score
som_size_x = 15
som_size_y = 15
som = MiniSom(som_size_x, som_size_y, input_len=X_scaled.shape[1], sigma=1.5, learning_rate=0.3)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 3000)

# Extract winning nodes and apply clustering
winning_nodes = np.array([som.winner(x) for x in X_scaled])
winning_node_features = np.apply_along_axis(lambda x: x[0] * som_size_y + x[1], 1, winning_nodes)

gmm = GaussianMixture(n_components=2, random_state=0)
gmm_labels = gmm.fit_predict(winning_node_features.reshape(-1, 1))

seasonality_feature['Cluster'] = gmm_labels

# Evaluate with Silhouette Score
sil_score = silhouette_score(X_scaled, gmm_labels)
print(f'Silhouette Score: {sil_score}')

# Plot with updated clustering
plt.figure(figsize=(12, 12))
plt.pcolor(som.distance_map().T, cmap='coolwarm')
plt.colorbar()

colors = ['red', 'blue', 'green', 'purple', 'orange']
for i, (x, y) in enumerate(winning_nodes):
    x_offset = np.random.uniform(-0.3, 0.3)
    y_offset = np.random.uniform(-0.3, 0.3)
    plt.text(x + 0.5 + x_offset, y + 0.5 + y_offset, seasonality_feature['PROVINCE'][i],
             ha='center', va='center', bbox=dict(facecolor=colors[seasonality_feature['Cluster'][i] % len(colors)], alpha=0.5), fontsize=8)

plt.title('SOM with GMM Clustering Based on Seasonality')
plt.show()