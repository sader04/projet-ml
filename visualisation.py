import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

print("GÉNÉRATION DES VISUALISATIONS POUR STREAMLIT")

#Créer un dossier pour les visualisations
os.makedirs("reports", exist_ok=True)

#Simuler/créer reconstruction_errors_distribution.png
print("\n1. Création: reconstruction_errors_distribution.png")

#Données simulées (ou charger depuis tes modèles)
np.random.seed(42)
n_normal = 950
n_anomaly = 50

#Erreurs de reconstruction simulées
errors_normal = np.random.exponential(scale=0.1, size=n_normal)
errors_anomaly = np.random.exponential(scale=0.8, size=n_anomaly)
all_errors = np.concatenate([errors_normal, errors_anomaly])
labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomaly)])

#Seuil simulé
threshold = 0.35

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#Histogramme
axes[0].hist(errors_normal, bins=40, alpha=0.6, label='Normal', color='blue', density=True)
axes[0].hist(errors_anomaly, bins=20, alpha=0.6, label='Anomalie', color='red', density=True)
axes[0].axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Seuil = {threshold:.2f}')
axes[0].set_xlabel('Erreur de reconstruction (MSE)')
axes[0].set_ylabel('Densité')
axes[0].set_title('Distribution des erreurs')
axes[0].legend()
axes[0].grid(alpha=0.3)

#Boxplot
axes[1].boxplot([errors_normal, errors_anomaly], labels=['Normal', 'Anomalie'])
axes[1].axhline(threshold, color='black', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_ylabel('Erreur MSE')
axes[1].set_title('Erreurs par classe')
axes[1].grid(alpha=0.3, axis='y')

#KDE
sns.kdeplot(errors_normal, ax=axes[2], label='Normal', color='blue', fill=True, alpha=0.3)
sns.kdeplot(errors_anomaly, ax=axes[2], label='Anomalie', color='red', fill=True, alpha=0.3)
axes[2].axvline(threshold, color='black', linestyle='--', linewidth=2, label='Seuil')
axes[2].set_xlabel('Erreur MSE')
axes[2].set_ylabel('Densité')
axes[2].set_title('Densité des erreurs')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.suptitle('Distribution des erreurs de reconstruction - Autoencodeur Dense', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('reconstruction_errors_distribution.png', dpi=150, bbox_inches='tight')
print("   ✅ reconstruction_errors_distribution.png sauvegardé")

#2. Créer latent_space_pca_analysis.png
print("\n2. Création: latent_space_pca_analysis.png")
fig2, ax = plt.subplots(figsize=(10, 8))

#Données simulées pour l'espace latent
latent_normal = np.random.multivariate_normal(
    mean=[0, 0], 
    cov=[[1, 0.3], [0.3, 1]], 
    size=n_normal
)
latent_anomaly = np.random.multivariate_normal(
    mean=[2.5, 2.5], 
    cov=[[1, -0.4], [-0.4, 1]], 
    size=n_anomaly
)

ax.scatter(latent_normal[:, 0], latent_normal[:, 1], 
          alpha=0.6, s=20, label='Normal', color='steelblue')
ax.scatter(latent_anomaly[:, 0], latent_anomaly[:, 1], 
          alpha=0.8, s=40, label='Anomalie', color='crimson', marker='^')

ax.set_xlabel('Composante latente 1', fontsize=12)
ax.set_ylabel('Composante latente 2', fontsize=12)
ax.set_title('Espace latent - Analyse PCA', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latent_space_pca_analysis.png', dpi=150, bbox_inches='tight')
print("   ✅ latent_space_pca_analysis.png sauvegardé")

#3. Créer cluster_error_distribution.png
print("\n3. Création: cluster_error_distribution.png")
fig3, ax = plt.subplots(figsize=(10, 6))

#Simuler 3 clusters
cluster_labels = np.random.choice([0, 1, 2], size=len(all_errors), p=[0.4, 0.4, 0.2])
clusters = [0, 1, 2]
colors = ['skyblue', 'lightgreen', 'salmon']

for cluster_id, color in zip(clusters, colors):
    cluster_errors = all_errors[cluster_labels == cluster_id]
    if len(cluster_errors) > 0:
        ax.hist(cluster_errors, bins=30, alpha=0.6, label=f'Cluster {cluster_id}', 
                color=color, density=True)

ax.set_xlabel('Erreur de reconstruction', fontsize=12)
ax.set_ylabel('Densité', fontsize=12)
ax.set_title('Distribution des erreurs par cluster', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_error_distribution.png', dpi=150, bbox_inches='tight')
print("   ✅ cluster_error_distribution.png sauvegardé")

#4. Vérifier/Créer model_comparison_results.csv
print("\n4. Vérification: model_comparison_results.csv")
if not os.path.exists('model_comparison_results.csv'):
    #Créer un exemple si le fichier n'existe pas
    example_data = {
        'Model': ['Autoencodeur Dense', 'Autoencodeur LSTM', 'Isolation Forest', 'One-Class SVM'],
        'Recall': [0.92, 0.88, 0.85, 0.80],
        'F2-Score': [0.87, 0.83, 0.79, 0.75],
        'Precision': [0.78, 0.75, 0.72, 0.68],
        'AUC-ROC': [0.95, 0.93, 0.90, 0.87],
        'AUC-PR': [0.89, 0.86, 0.82, 0.79],
        'TP': [46, 44, 42, 40],
        'FP': [13, 15, 16, 18],
        'FN': [4, 6, 8, 10],
        'TN': [937, 935, 934, 932]
    }
    
    pd.DataFrame(example_data).to_csv('model_comparison_results.csv', index=False)
    print("   ✅ model_comparison_results.csv créé (exemple)")
else:
    print("   ✅ model_comparison_results.csv existe déjà")

print("\n" + "=" * 60)
print("✅ TOUTES LES VISUALISATIONS ONT ÉTÉ GÉNÉRÉES !")
print("=" * 60)
print("- reconstruction_errors_distribution.png")
print("- latent_space_pca_analysis.png") 
print("- cluster_error_distribution.png")
print("- model_comparison_results.csv")
