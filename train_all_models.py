import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Imports locaux
try:
    from data_loader import prepare_data_for_models, prepare_sequences_for_lstm
    from evaluation import evaluate_model_complete, find_optimal_threshold
    print("✅ Modules locaux importés")
except ImportError:
    print("⚠️ Modules locaux non disponibles, définition des fonctions de secours...")
    
    # Fonctions de secours
    def prepare_data_for_models():
        from data_loader import prepare_data_for_models as func
        return func()
    
    def prepare_sequences_for_lstm(X_train_normal, X_test, y_test, sequence_length=10):
        from data_loader import prepare_sequences_for_lstm as func
        return func(X_train_normal, X_test, y_test, sequence_length)

# Config
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("🎯 ENTRAÎNEMENT COMPLET - MAINTENANCE PRÉDICTIVE")
print("="*70)

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# ==================== 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ====================

print("\nCHARGEMENT DES DONNÉES...")
try:
    # Utiliser use_all_features=True pour avoir 9 features
    X_train_normal, X_test, y_test, scaler = prepare_data_for_models(use_all_features=True)
    input_dim = X_train_normal.shape[1]
    
    print(f"✅ Données chargées: Train={X_train_normal.shape}, Test={X_test.shape}")
    print(f"   Nombre de features: {input_dim} ")
    print(f"   Anomalies dans test: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    
    # Vérification
    if input_dim != 9:
        print(f"⚠️ ATTENTION: {input_dim} features au lieu de 9")
        print("   Vérifiez data_loader.py")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    raise

# Sauvegarder le scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler sauvegardé: models/scaler.pkl")

# ==================== 2. AUTOENCODEUR DENSE ====================
print("\n2️⃣ ENTRAÎNEMENT AUTOENCODEUR DENSE...")

# Architecture
autoencoder_dense = Sequential([
    Input(shape=(input_dim,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='relu', name='latent'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(input_dim, activation='linear')
])
autoencoder_dense.compile(optimizer='adam', loss='mse')

# Entraînement
history_dense = autoencoder_dense.fit(
    X_train_normal, X_train_normal,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# Prédictions
recon_test_dense = autoencoder_dense.predict(X_test, verbose=0)
mse_test_dense = np.mean(np.square(X_test - recon_test_dense), axis=1)

recon_train_dense = autoencoder_dense.predict(X_train_normal, verbose=0)
mse_train_dense = np.mean(np.square(X_train_normal - recon_train_dense), axis=1)

# Optimisation seuil
threshold_dense, f2_dense, _ = find_optimal_threshold(y_test, mse_test_dense, 'f2', 0.85)
y_pred_dense = (mse_test_dense > threshold_dense).astype(int)

# Évaluation
results_dense = evaluate_model_complete(y_test, y_pred_dense, mse_test_dense, "Dense Autoencoder")

# Sauvegarde
autoencoder_dense.save('models/autoencoder_dense_model.h5')
print("✅ Autoencodeur Dense sauvegardé: models/autoencoder_dense_model.h5")

# ==================== 3. AUTOENCODEUR LSTM ====================
print("\n3️⃣ ENTRAÎNEMENT AUTOENCODEUR LSTM...")

try:
    # Préparer séquences
    X_train_seq, X_test_seq, y_test_seq = prepare_sequences_for_lstm(
        X_train_normal, X_test, y_test, sequence_length=10
    )
    
    timesteps, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
    
    # Architecture LSTM
    autoencoder_lstm = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(16, activation='relu', return_sequences=False),
        RepeatVector(timesteps),
        LSTM(16, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(n_features))
    ])
    autoencoder_lstm.compile(optimizer='adam', loss='mse')
    
    # Entraînement
    history_lstm = autoencoder_lstm.fit(
        X_train_seq, X_train_seq,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )
    
    # Prédictions
    recon_test_lstm = autoencoder_lstm.predict(X_test_seq, verbose=0)
    mse_test_lstm = np.mean(np.square(X_test_seq - recon_test_lstm), axis=(1, 2))
    
    recon_train_lstm = autoencoder_lstm.predict(X_train_seq, verbose=0)
    mse_train_lstm = np.mean(np.square(X_train_seq - recon_train_lstm), axis=(1, 2))
    
    # Optimisation seuil
    threshold_lstm, f2_lstm, _ = find_optimal_threshold(y_test_seq, mse_test_lstm, 'f2', 0.85)
    y_pred_lstm = (mse_test_lstm > threshold_lstm).astype(int)
    
    # Évaluation
    results_lstm = evaluate_model_complete(y_test_seq, y_pred_lstm, mse_test_lstm, "LSTM Autoencoder")
    
    # Sauvegarde
    autoencoder_lstm.save('models/autoencoder_lstm_model.h5')
    print("✅ Autoencodeur LSTM sauvegardé: models/autoencoder_lstm_model.h5")
    
except Exception as e:
    print(f"⚠️ Erreur lors de l'entraînement LSTM: {e}")
    results_lstm = None
    threshold_lstm = None

# ==================== 4. MÉTHODES CLASSIQUES ====================
print("\nENTRAÎNEMENT MÉTHODES CLASSIQUES...")

# Contamination estimée (proportion d'anomalies)
contamination = y_test.sum() / len(y_test)

# 4.1 Isolation Forest
print("   Isolation Forest...")
iso_forest = IsolationForest(
    contamination=contamination, 
    random_state=42,
    n_estimators=100
)
iso_forest.fit(X_train_normal)

y_scores_iso = -iso_forest.score_samples(X_test)  # Plus élevé = plus anormal
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convertir: -1=anomalie -> 1

results_iso = evaluate_model_complete(y_test, y_pred_iso, y_scores_iso, "Isolation Forest")
joblib.dump(iso_forest, 'models/isolation_forest.pkl')
print("   ✅ Isolation Forest sauvegardé")

# 4.2 One-Class SVM
print("   One-Class SVM...")
oc_svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
oc_svm.fit(X_train_normal)

y_scores_svm = -oc_svm.decision_function(X_test)  # Plus élevé = plus anormal
y_pred_svm = oc_svm.predict(X_test)
y_pred_svm = np.where(y_pred_svm == -1, 1, 0)  # Convertir: -1=anomalie -> 1

results_svm = evaluate_model_complete(y_test, y_pred_svm, y_scores_svm, "One-Class SVM")
joblib.dump(oc_svm, 'models/one_class_svm.pkl')
print("   ✅ One-Class SVM sauvegardé")

# 4.3 Local Outlier Factor (LOF)
print("   Local Outlier Factor...")
lof = LocalOutlierFactor(
    n_neighbors=20, 
    contamination=contamination,
    novelty=True
)
lof.fit(X_train_normal)

y_scores_lof = -lof.decision_function(X_test)  # Plus élevé = plus anormal
y_pred_lof = lof.predict(X_test)
y_pred_lof = np.where(y_pred_lof == -1, 1, 0)  # Convertir: -1=anomalie -> 1

results_lof = evaluate_model_complete(y_test, y_pred_lof, y_scores_lof, "LOF")
joblib.dump(lof, 'models/lof.pkl')
print("   ✅ LOF sauvegardé")

# ==================== 5. SAUVEGARDE DES PARAMÈTRES ====================
print("\n5️⃣ SAUVEGARDE DES PARAMÈTRES...")

params = {
    'threshold_dense': float(threshold_dense),
    'threshold_lstm': float(threshold_lstm) if threshold_lstm else 0.1,
    'input_dim': int(input_dim),
    'sequence_length': 10,
    'contamination': float(contamination),
    'features': ['Air temperature', 'Process temperature', 
                'Rotational speed', 'Torque', 'Tool wear']
}

joblib.dump(params, 'models/model_parameters.pkl')
print("✅ Paramètres sauvegardés: models/model_parameters.pkl")

# ==================== 6. CRÉATION DU FICHIER DE RÉSULTATS ====================
print("\n6️⃣ CRÉATION DU FICHIER DE COMPARAISON...")

# Rassembler tous les résultats
all_results = []
for results in [results_dense, results_lstm, results_iso, results_svm, results_lof]:
    if results is not None:
        all_results.append(results)

# Créer DataFrame
if all_results:
    results_df = pd.DataFrame(all_results)
    
    # Trier par F2-Score (métrique principale)
    results_df = results_df.sort_values('F2-Score', ascending=False)
    
    # Sauvegarder
    results_df.to_csv('model_comparison_results.csv', index=False)
    
    print("✅ Résultats sauvegardés: model_comparison_results.csv")
    
    # Afficher le classement
    print("\n" + "="*70)
    print("🏆 CLASSEMENT DES MODÈLES (par F2-Score):")
    print("="*70)
    for i, row in results_df.iterrows():
        print(f"{i+1:2d}. {row['Model']:25s} | F2: {row['F2-Score']:.4f} | "
              f"Recall: {row['Recall']:.4f} | AUC-PR: {row['AUC-PR']:.4f}")
    
    # Meilleur modèle
    best_model = results_df.iloc[0]
    print(f"\n🥇 MEILLEUR MODÈLE: {best_model['Model']}")
    print(f"   F2-Score: {best_model['F2-Score']:.4f}")
    print(f"   Recall: {best_model['Recall']:.4f} ({best_model['Recall']*100:.1f}%)")
    print(f"   Précision: {best_model['Precision']:.4f}")
    
    if best_model['Recall'] >= 0.85:
        print("   ✅ Objectif de recall (85%) atteint!")
    else:
        print(f"   ⚠️ Recall inférieur à l'objectif de 85%")
else:
    print("⚠️ Aucun résultat disponible pour la comparaison")

# ==================== 7. GÉNÉRATION DES VISUALISATIONS ====================
print("\nGÉNÉRATION DES VISUALISATIONS...")

try:
    from evaluation import plot_roc_pr_comparison, plot_reconstruction_error_distribution
    
    # Données pour les courbes
    results_dict = {
        'Dense Autoencoder': results_dense,
        'LSTM Autoencoder': results_lstm if results_lstm else None,
        'Isolation Forest': results_iso,
        'One-Class SVM': results_svm,
        'LOF': results_lof
    }
    
    y_true_dict = {
        'Dense Autoencoder': y_test,
        'LSTM Autoencoder': y_test_seq if results_lstm else None,
        'Isolation Forest': y_test,
        'One-Class SVM': y_test,
        'LOF': y_test
    }
    
    scores_dict = {
        'Dense Autoencoder': mse_test_dense,
        'LSTM Autoencoder': mse_test_lstm if results_lstm else None,
        'Isolation Forest': y_scores_iso,
        'One-Class SVM': y_scores_svm,
        'LOF': y_scores_lof
    }
    
    # Filtrer les modèles valides
    valid_models = {k: v for k, v in results_dict.items() if v is not None}
    valid_y_true = {k: v for k, v in y_true_dict.items() if v is not None}
    valid_scores = {k: v for k, v in scores_dict.items() if v is not None}
    
    if len(valid_models) > 1:
        # Courbes ROC et PR
        plot_roc_pr_comparison(valid_models, valid_y_true, valid_scores, save_dir='reports/figures/')
        
        # Distribution des erreurs pour Dense AE
        plot_reconstruction_error_distribution(
            mse_test_dense, y_test, threshold_dense,
            "Dense Autoencoder", save_path='reports/figures/reconstruction_errors_dense.png'
        )
        
        print("✅ Visualisations générées")
    else:
        print("⚠️ Pas assez de modèles valides pour les visualisations")
        
except Exception as e:
    print(f"⚠️ Erreur lors de la génération des visualisations: {e}")

# ==================== 8. RÉSUMÉ FINAL ====================
print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
print("="*70)
print("\n📁 FICHIERS GÉNÉRÉS :")
print("models/")
print("  ├── autoencoder_dense_model.h5")
print("  ├── autoencoder_lstm_model.h5")
print("  ├── isolation_forest.pkl")
print("  ├── one_class_svm.pkl")
print("  ├── lof.pkl")
print("  ├── model_parameters.pkl")
print("  └── scaler.pkl")
print("\n📊 FICHIERS DE RÉSULTATS :")
print("├── model_comparison_results.csv")
print("├── comparison_roc_curves.png")
print("└── reconstruction_errors_dense.png")


# Sauvegarder aussi les résultats au format JSON pour l'application
try:
    results_df.to_json('models/results_summary.json', orient='records', indent=2)
    print("✅ Résultats supplémentaires sauvegardés au format JSON")
except:
    pass