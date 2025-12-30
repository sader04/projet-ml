#pour charger et préparer les données pour l'entraînement des modèles
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

PROCESSED_DATA_DIR = "data/processed"

#FONCTIONS DE CHARGEMENT

def load_processed_data():
    """Charger les données prétraitées depuis l'EDA."""
    X = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "data_prepared.csv"))
    y = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "targets.csv"))
    
    print(f"Données chargées : X shape = {X.shape}, y shape = {y.shape}")
    return X, y


#Cette fonction est nécessaire pour l'app.py
def get_preprocessed_data(data_dir="data/processed"):
    #charge les données pré-traitées depuis les fichiers CSV.
    X_path = os.path.join(data_dir, "data_prepared.csv")
    y_path = os.path.join(data_dir, "targets.csv")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            f"Les fichiers de données n'existent pas dans {data_dir}. "
            f"Assurez-vous d'avoir exécuté le script d'EDA."
        )
    
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)
    
    print(f"✅ Données chargées depuis {data_dir}")
    print(f"   X shape: {X_df.shape}, y shape: {y_df.shape}")
    
    return X_df, y_df

#FONCTIONS DE PRÉPARATION POUR LES MODÈLES

def prepare_data_for_models(test_size=0.2, random_state=42, scale=True, use_all_features=True):
    #préparer les données pour l'entraînement des modèles.
    
    X_df, y_df = load_processed_data()
    
    if use_all_features:
        # Utiliser TOUTES les 9 features
        features = [
            'Air temperature', 
            'Process temperature', 
            'Rotational speed',
            'Torque', 
            'Tool wear',
            'Delta_T',     
            'Type_H',     
            'Type_L',       
            'Type_M'       
        ]
    else:
        # Utiliser seulement les 5 features continues (ancienne version)
        features = [
            'Air temperature', 
            'Process temperature', 
            'Rotational speed',
            'Torque', 
            'Tool wear'
        ]
    
    # Vérifier que toutes les features existent
    available_features = [f for f in features if f in X_df.columns]
    if len(available_features) != len(features):
        print(f"⚠️ Attention: {len(features) - len(available_features)} features manquantes")
        print(f"   Utilisation des features disponibles: {available_features}")
    
    X = X_df[available_features].values
    y = y_df['Machine failure'].values
    
    # Normalisation (CRUCIAL pour les autoencodeurs)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Séparer données normales et anomalies
    X_normal = X[y == 0]
    X_anomaly = X[y == 1]
    
    # Split train/test sur données normales
    X_train_normal, X_test_normal = train_test_split(
        X_normal, test_size=test_size, random_state=random_state
    )
    
    # Test set = normales + anomalies
    X_test = np.concatenate([X_test_normal, X_anomaly])
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_anomaly))])
    
    print(f"\nDonnées préparées pour les modèles :")
    print(f"   Train (normal only): {X_train_normal.shape}")
    print(f"   Test (normal+anomaly): {X_test.shape}")
    print(f"   Nombre de features: {X_train_normal.shape[1]}")
    print(f"   Anomalies dans test: {int(y_test.sum())} ({y_test.sum()/len(y_test)*100:.1f}%)")
    
    return X_train_normal, X_test, y_test, scaler


def create_sequences(data, sequence_length):
    #Crée des séquences temporelles pour LSTM
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


def prepare_sequences_for_lstm(X_train_normal, X_test, y_test, sequence_length=10):
    #Prépare les séquences pour l'autoencodeur LSTM.
    #Créer séquences
    X_train_seq = create_sequences(X_train_normal, sequence_length)
    X_test_seq = create_sequences(X_test, sequence_length)
    
    #Labels pour séquences (anomalie si au moins un point est anormal)
    y_test_seq = []
    for i in range(len(X_test_seq)):
        if 1 in y_test[i:i + sequence_length]:
            y_test_seq.append(1)
        else:
            y_test_seq.append(0)
    y_test_seq = np.array(y_test_seq)
    
    print(f"\nSéquences créées pour LSTM :")
    print(f"   Train sequences: {X_train_seq.shape}")
    print(f"   Test sequences: {X_test_seq.shape}")
    print(f"   Anomalies dans séquences: {int(y_test_seq.sum())} ({y_test_seq.sum()/len(y_test_seq)*100:.1f}%)")
    
    return X_train_seq, X_test_seq, y_test_seq


#FONCTIONS UTILITAIRES

def get_feature_names(use_all=True):
    #Retourne la liste des noms de features.
    if use_all:
        return [
            'Air temperature', 'Process temperature', 
            'Rotational speed', 'Torque', 'Tool wear',
            'Delta_T', 'Type_H', 'Type_L', 'Type_M'
        ]
    else:
        return [
            'Air temperature', 'Process temperature', 
            'Rotational speed', 'Torque', 'Tool wear'
        ]

def get_data_stats():
    #retourne des statistiques sur les données
    X_df, y_df = get_preprocessed_data()
    
    FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    
    stats = {
        'total_samples': len(X_df),
        'total_features': X_df.shape[1],
        'anomaly_count': int(y_df['Machine failure'].sum()),
        'anomaly_percentage': (y_df['Machine failure'].sum() / len(y_df) * 100),
        'failure_modes': {
            mode: int(y_df[mode].sum()) for mode in FAILURE_MODES
        }
    }
    
    return stats


def check_data_availability():
    #vérifie si les données sont disponibles
    try:
        X, y = get_preprocessed_data()
        return True, f"✅ Données disponibles ({len(X)} échantillons, {X.shape[1]} features)"
    except Exception as e:
        return False, f"❌ Erreur: {e}"


if __name__ == "__main__":
    #Test du module
    print("Test du module data_loader...")
    
    #Vérifier la disponibilité des données
    available, message = check_data_availability()
    print(message)
    
    if available:
        #Afficher les statistiques
        stats = get_data_stats()
        print(f"\nStatistiques des données :")
        print(f"   Total échantillons: {stats['total_samples']}")
        print(f"   Nombre de pannes: {stats['anomaly_count']} ({stats['anomaly_percentage']:.1f}%)")
        print(f"   Modes de panne:")
        for mode, count in stats['failure_modes'].items():
            print(f"     - {mode}: {count}")
        
        #Tester la préparation pour les modèles
        print("\nTest préparation pour modèles...")
        X_train, X_test, y_test, scaler = prepare_data_for_models()
        
        print("\nModule data_loader fonctionne correctement !")