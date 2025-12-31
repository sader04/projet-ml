# 📊 Projet ML - Détection d’Anomalies par Autoencodeurs en Maintenance Prédictive

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Keras](https://img.shields.io/badge/Keras-2.13-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B)

## 📌 Résumé du projet
Ce projet vise à mettre en œuvre et comparer différentes approches de détection d’anomalies dans un contexte de maintenance prédictive industrielle. L'accent est mis sur l'utilisation des **autoencodeurs** (réseaux de neurones non supervisés) pour apprendre les comportements normaux et détecter les déviations anormales dans les données de capteurs.

---

## 🎯 Objectifs principaux
1. **Modélisation** : Implémenter et comparer des modèles d'autoencodeurs (Dense et LSTM) pour la détection d'anomalies.
2. **Benchmark** : Comparer ces approches avec des méthodes classiques (Isolation Forest, One-Class SVM, LOF).
3. **Clustering** : Réaliser du clustering dans l'espace latent pour identifier des régimes de fonctionnement.
4. **Visualisation** : Analyser et interpréter l'espace latent via PCA et t-SNE.
5. **Déploiement** : Développer un tableau de bord interactif Streamlit pour une utilisation opérationnelle.

---

## 📂 Structure du dépôt
```
projet-ml/
├── data/ processed                         
│   ├── data_prepared.csv          # Données préparées pour l'entraînement
│   ├── data_raw_with_deltas.csv   # Données brutes avec deltas
│   └── targets.csv                # Labels cibles
│
├── models/                        # Modèles sauvegardés
├── reports/                       # Figures et analyses générées
│
├── app.py                         # Application principale Streamlit
├── data_loader.py                 # Chargement des données
├── data_preprocessing.py          # Prétraitement des données
├── evaluation.py                  # Évaluation des modèles
├── train_all_models.py            # Script d'entraînement
├── visualisation.py               # Fonctions de visualisation
│
├── requirements.txt               # Dépendances Python
├── README.md                      # Ce fichier
│
└──  model_comparison_results.csv   # Résultats comparatifs

```

---

## 📊 Jeu de données
- **Nom :** AI4I 2020 Predictive Maintenance Dataset  
- **Source :** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Caractéristiques :**
  - 10 000 observations, 14 variables
  - Données synthétiques simulées (capteurs, types de produits, modes de panne)
  - Taux de panne global : 3.39 % (problème déséquilibré)
  - Modes de défaillance : TWF, HDF, PWF, OSF, RNF

---

## 🧠 Méthodologie

### 1. **Analyse exploratoire (EDA)**

### 2. **Modélisation**

### 3. **Clustering dans l'espace latent**

### 4. **Visualisation**

### 5. **Évaluation**

---

## 🚀 Résultats clés

- Les autoencodeurs surpassent nettement les méthodes classiques
- L'autoencodeur LSTM est le plus performant grâce à sa capacité à capturer les dépendances temporelles
- L'espace latent permet une séparation naturelle des régimes de fonctionnement
- DBSCAN identifie les anomalies les plus extrêmes avec une grande confiance

---

## 🎨 Tableau de bord Streamlit
**Fonctionnalités principales :**
- Exploration interactive des données
- Comparaison des performances des modèles
- Visualisation des clusters dans l'espace latent
- Prédiction en temps réel via interface utilisateur
- Graphiques interactifs (scatter plots, distributions, heatmaps)

**Accès :** Lancer `streamlit run app.py` après installation

---

## 🛠 Technologies utilisées
- **Langage :** Python 
- **ML/DL :** TensorFlow/Keras, Scikit-learn
- **Data :** Pandas, NumPy
- **Visualisation :** Matplotlib, Seaborn, Plotly
- **Interface :** Streamlit

---

## 🚀 Installation et exécution COMPLÈTE

### 1. Cloner le dépôt
```bash
git clone https://github.com/sader04/projet-ml.git
cd PROJET-ML
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. **Workflow complet d'exécution** (nécessaire pour `app.py`)

#### Étape 1 : Préparation des données
```bash
python data_preprocessing.py
```
*Crée : `data_prepared.csv`, `targets.csv`*

#### Étape 2 : Chargement et préparation des datasets
```bash
python data_loader.py
```
*Importe les données prétraitées et prépare les splits train/test*

#### Étape 3 : Entraînement de tous les modèles
```bash
python train_all_models.py
```
*Entraîne :*
- *Autoencodeur Dense*
- *Autoencodeur LSTM* 
- *Isolation Forest*
- *One-Class SVM*
- *Local Outlier Factor*

#### Étape 4 : Évaluation des modèles
```bash
python evaluation.py
```
*Génère :*
- *`model_comparison_results.csv`*
- *Courbes ROC*
- *Métriques de performance*

#### Étape 5 : Génération des visualisations
```bash
python visualisation.py
```
*Crée les figures PNG :*
- *`comparison_roc_curves.png`*
- *`reconstruction_errors_dense.png`*
- *`latent_space_pca_analysis.png`*
- *`cluster_error_distribution.png`*

#### Étape 6 : Lancement du dashboard Streamlit
```bash
streamlit run app.py
```
*L'application `app.py` utilise :*
- *`data_loader.py` pour charger les données*
- *`evaluation.py` pour les métriques*
- *`visualisation.py` pour les graphiques*
- *Les modèles entraînés dans `/models/`*
- *Les visualisations générées dans `reports/figures`*

---

## ⚠️ Notes importantes

### Dépendances entre fichiers
- `app.py` **dépend** de tous les autres fichiers Python pour fonctionner correctement
- L'ordre d'exécution **doit être respecté** :
  1. `data_preprocessing.py`
  2. `data_loader.py`  
  3. `train_all_models.py`
  4. `evaluation.py`
  5. `visualisation.py`
  6. `app.py`

### Fichiers générés nécessaires
Pour que `app.py` fonctionne, les fichiers suivants doivent exister :
- `data_prepared.csv` (après `data_preprocessing.py`)
- Modèles entraînés dans `/models/` (après `train_all_models.py`)
- Visualisations PNG (après `visualisation.py`)
- `model_comparison_results.csv` (après `evaluation.py`)

---


## 🔮 Perspectives d'amélioration
- Intégration d'API pour flux temps réel (FastAPI)
- Système d'alertes automatisées (email/SMS)
- Archivage des prédictions en base de données
- Déploiement conteneurisé (Docker)
- Authentification utilisateur
- Adaptation à des données réelles industrielles

---

## 👥 Équipe
- **CHATBA Abir**
- **CHBIHI Doha**
- **DERBANI Salwa**
- **MAZOUZ Nour**

---

## 📄 Licence
Projet académique réalisé dans le cadre du cours du Machine Learning à l'Ecole Centrale Casablanca.  
Jeu de données : AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository).
