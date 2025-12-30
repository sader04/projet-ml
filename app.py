import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import joblib
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

#Configuration de la page
st.set_page_config(
    page_title="Maintenance Prédictive - Dashboard",
    page_icon="🔧",
    layout="wide"
)

#Style CSS 
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

#FONCTIONS DE CHARGEMENT DES MODÈLES

@st.cache_resource
def load_autoencoders_only():
    """Charge uniquement les autoencodeurs."""
    models_dict = {}
    
    #Charger les paramètres
    params_path = "models/model_parameters.pkl"
    if os.path.exists(params_path):
        try:
            with open(params_path, "rb") as f:
                models_dict['params'] = pickle.load(f)
                # st.sidebar.success("✅ Paramètres chargés")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Paramètres non chargés: {e}")
            models_dict['params'] = {'threshold_dense': 0.01, 'threshold_lstm': 0.01}
    
    #Charger le scaler (CRITIQUE pour les autoencodeurs)
    scaler_path = "models/scaler.pkl"
    if os.path.exists(scaler_path):
        try:
            #Essayer joblib d'abord
            import joblib
            models_dict['scaler'] = joblib.load(scaler_path)
            # st.sidebar.success("✅ Scaler chargé avec joblib")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur chargement joblib: {e}")
            try:
                #Fallback: essayer pickle
                with open(scaler_path, "rb") as f:
                    models_dict['scaler'] = pickle.load(f)
                #st.sidebar.success("✅ Scaler chargé avec pickle")
            except Exception as e2:
                st.sidebar.error(f"❌ Erreur chargement scaler: {e2}")
    else:
        st.sidebar.warning("⚠️ Scaler non trouvé, création d'urgence")
        models_dict['scaler'] = create_emergency_scaler()
    
    #Autoencodeur Dense
    dense_path = "models/autoencoder_dense_model.h5"
    if os.path.exists(dense_path):
        try:
            models_dict['dense'] = load_model(dense_path)
            # st.sidebar.success("✅ Autoencodeur Dense chargé")
        except Exception as e:
            st.sidebar.error(f"❌ Autoencodeur Dense non chargé: {e}")
    
    #Autoencodeur LSTM
    lstm_path = "models/autoencoder_lstm_model.h5"
    if os.path.exists(lstm_path):
        try:
            models_dict['lstm'] = load_model(lstm_path)
            # st.sidebar.success("✅ Autoencodeur LSTM chargé")
        except Exception as e:
            st.sidebar.error(f"❌ Autoencodeur LSTM non chargé: {e}")
    
    return models_dict
#FONCTIONS DE PRÉDICTION

def prepare_features_for_prediction(air_temp, process_temp, rotational_speed, torque, tool_wear, product_type):
    """Prépare les features pour la prédiction avec toutes les 9 features."""
    
    #Calculer Delta_T 
    delta_t = process_temp - air_temp
    
    #Encoder le type de produit (one-hot encoding comme dans l'entraînement)
    type_h = 1 if product_type == 'H' else 0
    type_l = 1 if product_type == 'L' else 0
    type_m = 1 if product_type == 'M' else 0

    features = np.array([[
        air_temp,           
        process_temp,       
        rotational_speed,   
        torque,            
        tool_wear,         
        delta_t,           
        type_h,            
        type_l,            
        type_m             
    ]])
    return features

def predict_with_dense_ae(features, models_dict):
    #Prédiction avec l'autoencodeur dense
    if 'dense' not in models_dict or 'scaler' not in models_dict:
        return None
    
    try:
        #Normaliser
        scaler = models_dict['scaler']
        features_scaled = scaler.transform(features)
        
        #Prédiction
        autoencoder = models_dict['dense']
        reconstruction = autoencoder.predict(features_scaled, verbose=0)
        
        #Calculer MSE
        mse = np.mean(np.square(features_scaled - reconstruction))
        
        #Seuil
        threshold = models_dict['params']['threshold_dense']
        
        return {
            'mse': mse,
            'threshold': threshold,
            'is_anomaly': mse > threshold,
            'anomaly_score': mse,
            'reconstruction': reconstruction,
            'features_scaled': features_scaled
        }
    except Exception as e:
        st.error(f"Erreur prédiction dense AE: {e}")
        return None

def predict_with_lstm_ae(features, models_dict):
    #Prédiction avec l'autoencodeur LSTM.
    if 'lstm' not in models_dict or 'scaler' not in models_dict:
        return None
    
    try:
        #Normaliser
        scaler = models_dict['scaler']
        features_scaled = scaler.transform(features)
        
        #Créer séquence 
        sequence_length = 10
        features_seq = np.tile(features_scaled, (sequence_length, 1))
        features_seq = features_seq.reshape(1, sequence_length, features_scaled.shape[1])
        
        #Prédiction
        lstm_ae = models_dict['lstm']
        reconstruction = lstm_ae.predict(features_seq, verbose=0)
        
        #Calculer MSE
        mse = np.mean(np.square(features_seq - reconstruction))
        
        #Seuil
        threshold = models_dict['params']['threshold_lstm']
        
        return {
            'mse': mse,
            'threshold': threshold,
            'is_anomaly': mse > threshold,
            'anomaly_score': mse,
            'reconstruction': reconstruction,
            'features_seq': features_seq
        }
    except Exception as e:
        st.error(f"Erreur prédiction LSTM AE: {e}")
        return None

#FONCTIONS UTILITAIRES

def get_preprocessed_data_local(data_dir="data/processed"):
    #Fonction de secours si le module n'est pas disponible
    try:
        X_path = os.path.join(data_dir, "data_prepared.csv")
        y_path = os.path.join(data_dir, "targets.csv")
        
        if os.path.exists(X_path) and os.path.exists(y_path):
            X = pd.read_csv(X_path)
            y = pd.read_csv(y_path)
            return X, y
        else:
            return None, None
    except Exception as e:
        return None, None

#Essayer d'importer le module data_loader
try:
    from data_loader import get_preprocessed_data
    st.sidebar.success("✅ Modules du projet importés")
except ImportError as e:
    st.sidebar.warning(f"⚠️ data_loader non disponible : {e}")
    #Utiliser la fonction locale
    get_preprocessed_data = get_preprocessed_data_local

#Charger tous les modèles au démarrage
models_dict = load_autoencoders_only()

#Titre principal
st.markdown("""
<div class="main-header">
    <h1>🔧 Dashboard de Maintenance Prédictive</h1>
    <p>Détection d'anomalies avec Autoencodeurs et Machine Learning</p>
</div>
""", unsafe_allow_html=True)

#Barre latérale
st.sidebar.title("🔧 Navigation")
section = st.sidebar.radio(
    "Sélectionnez une section :",
    ["🏠 Accueil", 
     "📊 Exploration des Données", 
     "🤖 Performance des modèles",
     "🎯 Prédiction en Temps Réel",
     "📈 Visualisations"]
)


#Afficher les autoencodeurs disponibles
st.sidebar.markdown("---")
st.sidebar.markdown("**🧠 Autoencodeurs :**")

if 'dense' in models_dict:
    st.sidebar.markdown("• ✅ Autoencodeur Dense")
if 'lstm' in models_dict:
    st.sidebar.markdown("• ✅ Autoencodeur LSTM")

st.sidebar.markdown("---")
st.sidebar.info("""
**Dataset :** AI4I 2020 Predictive Maintenance
**Techniques :**
- Autoencodeur Dense
- Autoencodeur LSTM
""")

#SECTION PRÉDICTION EN TEMPS RÉEL
if section == "🎯 Prédiction en Temps Réel":
    st.subheader("🎯 Détection d'Anomalie en Temps Réel")
    
    st.info("""
    **💡 Mode d'emploi :**
    1. Saisissez les valeurs des capteurs dans les champs ci-dessous
    2. Sélectionnez le modèle à utiliser pour la détection
    3. Cliquez sur "Détecter Anomalie" pour analyser l'état de la machine
    4. Visualisez le score d'anomalie et la décision
    """)
    
    #Formulaire de saisie
    st.markdown("### 📝 Paramètres de la Machine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        air_temp = st.slider(
            "🌡️ Température Air [K]", 
            min_value=290.0, 
            max_value=310.0, 
            value=298.1, 
            step=0.1
        )
        
        process_temp = st.slider(
            "🔥 Température Processus [K]", 
            min_value=300.0, 
            max_value=320.0, 
            value=308.6, 
            step=0.1
        )
        
        rotational_speed = st.slider(
            "⚙️ Vitesse de Rotation [rpm]", 
            min_value=1000, 
            max_value=3000, 
            value=1551, 
            step=10
        )
    
    with col2:
        torque = st.slider(
            "🔧 Couple [Nm]", 
            min_value=0.0, 
            max_value=100.0, 
            value=42.8, 
            step=0.1
        )
        
        tool_wear = st.slider(
            "🛠️ Usure Outil [min]", 
            min_value=0, 
            max_value=300, 
            value=0, 
            step=1
        )
        
        product_type = st.selectbox(
            "📦 Type de Produit", 
            options=['L', 'M', 'H'],
            index=1
        )
    
    #Sélection du modèle
    st.markdown("### 🤖 Sélection du Modèle")

    #Liste des modèles disponibles
    model_options = []

    if 'dense' in models_dict:
        model_options.append("Autoencodeur Dense")

    if 'lstm' in models_dict:
        model_options.append("Autoencodeur LSTM")


    if model_options:
        selected_model = st.selectbox(
            "Choisissez le modèle de détection :",
            model_options,
            help="Sélectionnez le modèle entraîné à utiliser pour la détection"
        )
        
        #Afficher les détails du modèle sélectionné
        if selected_model == "Autoencodeur Dense" and 'params' in models_dict:
            threshold = models_dict['params'].get('threshold_dense', 'Non disponible')
            st.info(f"**Seuil de détection :** {threshold:.6f} (99ème percentile)")
        
        elif selected_model == "Autoencodeur LSTM" and 'params' in models_dict:
            threshold = models_dict['params'].get('threshold_lstm', 'Non disponible')
            st.info(f"**Seuil de détection :** {threshold:.6f} (99ème percentile)")
        
        #Vérification des prérequis pour la prédiction
        if 'scaler' not in models_dict:
            st.error("❌ Le scaler n'est pas chargé ! Impossible de faire des prédictions.")
            st.stop()

        if 'dense' not in models_dict and 'lstm' not in models_dict:
            st.error("❌ Aucun autoencodeur n'est chargé !")
            st.info("Veuillez d'abord entraîner les modèles avec : python train_all_models.py")
            st.stop()

        
        #Bouton de prédiction
        if st.button("🔍 Détecter Anomalie", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours avec le modèle..."):
                try:
                    #Préparer les features
                    features = prepare_features_for_prediction(air_temp, process_temp, rotational_speed, torque, tool_wear, product_type)
                    
                    #Variables pour stocker les résultats
                    result = None
                    model_name = ""
                    
                    #Appeler la fonction de prédiction appropriée
                    if selected_model == "Autoencodeur Dense":
                        result = predict_with_dense_ae(features, models_dict)
                        model_name = "Autoencodeur Dense"
                    
                    elif selected_model == "Autoencodeur LSTM":
                        result = predict_with_lstm_ae(features, models_dict)
                        model_name = "Autoencodeur LSTM"
                    
                    
                    #Afficher les résultats
                    if result:
                        st.markdown("---")
                        st.markdown("### 📊 Résultat de l'Analyse")
                        
                        #Métriques
                        col_result1, col_result2, col_result3 = st.columns(3)
                        
                        with col_result1:
                            if result['is_anomaly']:
                                st.error("🔴 **ANOMALIE DÉTECTÉE**")
                            else:
                                st.success("🟢 **FONCTIONNEMENT NORMAL**")
                        
                        with col_result2:
                            if 'mse' in result:
                                st.metric(
                                    "Erreur de Reconstruction (MSE)",
                                    f"{result['mse']:.6f}",
                                    delta=f"{((result['mse']/result.get('threshold', 1) - 1) * 100):+.1f}% vs seuil" if 'threshold' in result else ""
                                )
                            elif 'score' in result:
                                st.metric(
                                    "Score d'anomalie",
                                    f"{result['score']:.4f}",
                                    delta="Plus bas = plus anormal"
                                )
                            elif 'anomaly_score' in result:
                                st.metric(
                                    "Score d'anomalie",
                                    f"{result['anomaly_score']:.4f}",
                                    delta="Plus élevé = plus anormal"
                                )
                        
                        with col_result3:
                            if 'threshold' in result:
                                st.metric(
                                    "Seuil de détection",
                                    f"{result['threshold']:.6f}",
                                    delta="99ème percentile"
                                )
                        
                        #Jauge d'anomalie pour les autoencodeurs
                        if 'mse' in result and 'threshold' in result:
                            st.markdown("### 📈 Niveau d'Anomalie")
                            
                            #Calculer le pourcentage d'anomalie
                            anomaly_percentage = min((result['mse'] / result['threshold']) * 100, 200)
                            
                            fig, ax = plt.subplots(figsize=(10, 2))
                            
                            #Barre de progression
                            ax.barh(['Anomalie'], [100], color='lightgreen', height=0.5)
                            ax.barh(['Anomalie'], [min(anomaly_percentage, 100)], 
                                color='red' if result['is_anomaly'] else 'green', 
                                height=0.5)
                            
                            #Ligne du seuil
                            ax.axvline(x=100, color='black', linestyle='--', alpha=0.7, linewidth=2)
                            
                            ax.set_xlim(0, 200)
                            ax.set_xlabel('Score d\'anomalie (%)')
                            ax.set_title(f"Score: {anomaly_percentage:.1f}% - Seuil: 100%")
                            
                            #Zones de couleur
                            ax.axvspan(0, 100, alpha=0.1, color='green')
                            ax.axvspan(100, 150, alpha=0.1, color='yellow')
                            ax.axvspan(150, 200, alpha=0.1, color='red')
                            
                            st.pyplot(fig)
                        
                        #Recommandations
                        st.markdown("---")
                        st.markdown("### 💡 Recommandations")
                        
                        if result['is_anomaly']:
                            # Calculer Delta_T
                            delta_t = process_temp - air_temp
                            
                            st.warning("""
                            **⚠️ ANOMALIE DÉTECTÉE - Actions recommandées :**
                            1. 🔍 **Inspection immédiate** de l'équipement
                            2. 📊 **Vérifier les logs** des dernières heures
                            3. 🛠️ **Maintenance préventive** à planifier
                            4. 📞 **Alerter l'équipe** de maintenance
                            """)
                            
                            #Détection de type de panne probable
                            failure_types = []
                            
                            if delta_t > 10:
                                failure_types.append("HDF (Heat Dissipation Failure)")
                            
                            if tool_wear > 200:
                                failure_types.append("TWF (Tool Wear Failure)")
                            
                            if rotational_speed < 1380:
                                failure_types.append("Problème de vitesse")
                            
                            if torque > 60:
                                failure_types.append("OSF (Overstrain Failure)")
                            
                            if failure_types:
                                st.error(f"**Panne(s) probable(s) :** {', '.join(failure_types)}")
                        
                        else:
                            st.success("""
                            **✅ FONCTIONNEMENT NORMAL - Actions recommandées :**
                            1. ✅ **Continuer le monitoring** régulier
                            2. 📊 **Archiver ces mesures** pour analyse future
                            3. 🔄 **Maintenance préventive** selon le planning habituel
                            """)
                        
                        #Détails techniques
                        with st.expander("🔬 Détails Techniques"):
                            st.markdown(f"**Modèle utilisé :** {model_name}")
                            
                            if 'mse' in result:
                                st.markdown(f"**Erreur MSE :** {result['mse']:.6f}")
                            
                            if 'threshold' in result:
                                st.markdown(f"**Seuil :** {result['threshold']:.6f}")
                            
                            if 'features_scaled' in result:
                                st.markdown("**Features normalisées :**")
                                st.code(str(result['features_scaled']))
                            
                            #Afficher les paramètres
                            st.markdown("**Paramètres d'entrée :**")

                            #Créer un DataFrame avec les bonnes colonnes
                            data = {
                                'Paramètre': ['Air temperature [K]', 'Process temperature [K]', 
                                            'Rotational speed [rpm]', 'Torque [Nm]', 
                                            'Tool wear [min]', 'Product Type', 'Delta_T [K]'],
                                'Valeur': [f"{air_temp:.2f}", f"{process_temp:.2f}", 
                                        f"{rotational_speed:.0f}", f"{torque:.2f}", 
                                        f"{tool_wear:.0f}", product_type, 
                                        f"{process_temp - air_temp:.2f}"]
                            }

                            df_params = pd.DataFrame(data)
                            st.dataframe(df_params, width='stretch', hide_index=True)
                    else:
                        st.error("❌ Erreur lors de la prédiction. Vérifiez les modèles.")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction : {str(e)}")
                    st.info("""
                    **Dépannage :**
                    1. Vérifiez que le modèle est correctement chargé
                    2. Assurez-vous que les dimensions des données sont correctes
                    3. Vérifiez que le scaler correspond aux données d'entrée
                    """)
        


    else:  # if not model_options:
        st.error("❌ Aucun autoencodeur disponible !")
        st.info("""
        Pour entraîner les autoencodeurs, exécutez :
        ```bash
        python train_all_models.py
        ```
        
        Assurez-vous que ces fichiers existent :
        - `models/autoencoder_dense_model.h5`
        - `models/autoencoder_lstm_model.h5`
        - `models/model_parameters.pkl`
        - `models/scaler.pkl`
        """)
        st.stop()

elif section == "🏠 Accueil":
    st.subheader("🎯 Aperçu du Projet")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Contexte
        Ce projet de **Maintenance Prédictive** vise à détecter les anomalies dans 
        les équipements industriels avant qu'elles ne conduisent à des pannes coûteuses.
        
        ### Objectifs
        1. **Identifier les modes de défaillance** spécifiques
        2. **Réduire les coûts** de maintenance non planifiée
        3. **Prolonger la durée de vie** des équipements
        
        ### Approche Technique
        - **Apprentissage non supervisé** : Modèles entraînés uniquement sur les données normales
        - **Autoencodeurs** : Reconstruction pour détecter les déviations
        - **Clustering** : Analyse de l'espace latent pour identifier les régimes
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Variables du Dataset
        **Capteurs :**
        - 🌡️ Température de l'air
        - 🔥 Température du processus
        - ⚙️ Vitesse de rotation
        - 🔧 Couple
        - 🛠️ Usure de l'outil
        
        **Types de produit :**
        - L (Low quality)
        - M (Medium quality) 
        - H (High quality)
        
        **Modes de panne :**
        - TWF, HDF, PWF, OSF, RNF
        """)
    
    #Chargement rapide des données
    st.markdown("---")
    st.subheader("📊 Chargement des Données")
    
    if st.button("🔄 Charger les données"):
        with st.spinner("Chargement en cours..."):
            X, y = get_preprocessed_data()
            
            if X is not None and y is not None:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Observations", f"{len(X):,}")
                
                with col2:
                    failures = y['Machine failure'].sum()
                    st.metric("Pannes", f"{failures:,}", 
                             f"{(failures/len(y)*100):.1f}%")
                
                with col3:
                    st.metric("Features", X.shape[1])
                
                with col4:
                    st.metric("Modes de panne", 5)
                
                #Aperçu des données
                st.markdown("**Aperçu des données :**")
                tab1, tab2 = st.tabs(["Features", "Targets"])
                with tab1:
                    X_display = X.head(10).copy()
                    for col in X_display.select_dtypes(include=['object', 'category']).columns:
                        X_display[col] = X_display[col].astype(str)
                    st.dataframe(X_display, width='stretch')
                    
                with tab2:
                    y_display = y.head(10).copy()
                    for col in y_display.select_dtypes(include=['object', 'category']).columns:
                        y_display[col] = y_display[col].astype(str)
                    st.dataframe(y_display, width='stretch')
            else:
                st.error("Impossible de charger les données")

elif section == "📊 Exploration des Données":
    st.subheader("📊 Exploration et Analyse des Données")
    
    #Charger les données
    X, y = get_preprocessed_data()
    
    if X is not None and y is not None:
        #Options d'exploration
        analysis_type = st.selectbox(
            "Type d'analyse :",
            ["Distribution des variables", 
             "Corrélations", 
             "Analyse par type de produit",
             "Fréquence des pannes"]
        )
        
        if analysis_type == "Distribution des variables":
            st.markdown("### Distribution des Variables Continues")
            
            variables = ['Air temperature', 'Process temperature', 
                        'Rotational speed', 'Torque', 'Tool wear']
            
            selected_var = st.selectbox("Sélectionnez une variable :", variables)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=X, x=selected_var, kde=True, ax=ax)
                ax.set_title(f"Distribution de {selected_var}")
                st.pyplot(fig)
            
            with col2:
                st.markdown("**Statistiques :**")
                stats = X[selected_var].describe()
                for stat, value in stats.items():
                    st.metric(stat.capitalize(), f"{value:.2f}")
        
        elif analysis_type == "Corrélations":
            st.markdown("### Matrice de Corrélation")
            
            #Filtrer les colonnes numériques
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                corr_matrix = X[numeric_cols].corr()
                
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                           center=0, ax=ax, square=True)
                ax.set_title("Matrice de Corrélation")
                st.pyplot(fig)
            else:
                st.warning("Pas assez de variables numériques pour la corrélation")
        
        elif analysis_type == "Analyse par type de produit":
            st.markdown("### Répartition par Type de Produit")
            
            #Compter les types de produit
            if 'Type_L' in X.columns and 'Type_M' in X.columns and 'Type_H' in X.columns:
                type_counts = {
                    'L': X['Type_L'].sum(),
                    'M': X['Type_M'].sum(),
                    'H': X['Type_H'].sum()
                }
                
                df_types = pd.DataFrame({
                    'Type': list(type_counts.keys()),
                    'Count': list(type_counts.values())
                })
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                #Pie chart
                colors = ['#FF6B6B', '#FFD166', '#06D6A0']
                ax1.pie(df_types['Count'], labels=df_types['Type'], 
                       autopct='%1.1f%%', colors=colors, startangle=90)
                ax1.set_title("Répartition des types de produit")
                
                #Bar chart
                bars = ax2.bar(df_types['Type'], df_types['Count'], color=colors)
                ax2.set_title("Nombre par type de produit")
                ax2.set_ylabel("Nombre")
                
                #Ajouter les valeurs sur les barres
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom')
                
                st.pyplot(fig)
        
        elif analysis_type == "Fréquence des pannes":
            st.markdown("### Fréquence des Modes de Panne")
            
            failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
            failure_counts = y[failure_modes].sum().sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(failure_counts.index, failure_counts.values, 
                         color=['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C'])
            ax.set_title("Fréquence des modes de panne")
            ax.set_ylabel("Nombre d'occurrences")
            ax.set_xlabel("Type de panne")
            
            #Ajouter les valeurs
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            #Description des modes de panne
            st.markdown("**Description des modes de panne :**")
            st.info("""
            - **TWF** : Tool Wear Failure - Panne due à l'usure de l'outil
            - **HDF** : Heat Dissipation Failure - Panne de dissipation thermique
            - **PWF** : Power Failure - Panne de puissance
            - **OSF** : Overstrain Failure - Panne due à la surcharge
            - **RNF** : Random Failure - Panne aléatoire
            """)
    else:
        st.error("Les données ne sont pas disponibles. Veuillez d'abord exécuter le script d'EDA.")

elif section == "🤖 Performance des modèles":
    st.subheader("🤖 Comparaison des Modèles de Détection")
    
    #Vérifier si les résultats existent
    results_file = "model_comparison_results.csv"
    
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
        
        #Afficher le tableau des résultats
        st.markdown("### 📊 Performances des Modèles")
        
        #Trier par F2-Score (métrique principale)
        results_sorted = results_df.sort_values('F2-Score', ascending=False)
        
        #Formater l'affichage
        display_cols = ['Model', 'Recall', 'F2-Score', 'Precision', 
                       'AUC-ROC', 'AUC-PR', 'TP', 'FP', 'FN', 'TN']
        
        #Filtrer les colonnes existantes
        display_cols = [col for col in display_cols if col in results_sorted.columns]

        display_df = results_sorted[display_cols].fillna(0)
        st.dataframe(
            display_df.style.format({
                'Recall': '{:.4f}',
                'F2-Score': '{:.4f}',
                'Precision': '{:.4f}',
                'AUC-ROC': '{:.4f}',
                'AUC-PR': '{:.4f}'
            }).background_gradient(subset=['Recall', 'F2-Score'], cmap='RdYlGn'),
            width='stretch'
        )

        #Afficher le meilleur modèle
        best_model = results_sorted.iloc[0]
        
        st.markdown("---")
        st.markdown(f"### 🏆 Meilleur Modèle : **{best_model['Model']}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("F2-Score", f"{best_model['F2-Score']:.4f}")
        
        with col2:
            st.metric("Recall", f"{best_model['Recall']:.4f}")
        
        with col3:
            if best_model['Recall'] > 0.85:
                st.success("✅ Objectif atteint (>85%)")
            else:
                st.warning(f"⚠️ Recall à améliorer")
        
        #Recommandation
        if best_model['Recall'] >= 0.85:
            st.success(f"""
            **✅ Excellent !** Le modèle {best_model['Model']} détecte **{best_model['Recall']*100:.1f}%** des anomalies.
            Cette performance est idéale pour la maintenance prédictive où détecter les pannes
            est plus important que d'éviter les fausses alarmes.
            """)
        else:
            st.warning(f"""
            **⚠️ Attention !** Le Recall est de {best_model['Recall']*100:.1f}%.
            En maintenance prédictive, manquer une panne coûte généralement beaucoup plus cher
            qu'une fausse alarme.
            """)
        
        #Graphiques de comparaison
        st.markdown("---")
        st.markdown("### 📈 Visualisation des Performances")
        
        if os.path.exists("comparison_roc_curves.png"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image("comparison_roc_curves.png", 
                        caption="Courbes ROC - Tous les modèles",
                        width='stretch')
            
            # with col2:
            #     if os.path.exists("comparison_pr_curves.png"):
            #         st.image("comparison_pr_curves.png", 
            #                 caption="Courbes Precision-Recall",
            #                 width='stretch')
        
    else:
        st.warning("""
        ⚠️ **Les résultats des modèles ne sont pas disponibles.**
        
        Pour générer ces résultats, veuillez exécuter le script d'entraînement des modèles :
        
        ```bash
        python train_all_models.py
        ```
        
        Ou exécutez le notebook principal pour entraîner tous les modèles et générer les résultats.
        """)

elif section == "📈 Visualisations":
    st.subheader("📈 Visualisations Avancées")
    
    #Charger les données
    X, y = get_preprocessed_data()
    
    if X is not None and y is not None:
        # Options de visualisation
        viz_type = st.selectbox(
            "Type de visualisation :",
            ["Scatter plot interactif", 
             "Distribution conditionnelle", 
             "Analyse de clusters",
             "Erreurs de reconstruction"]
        )
        
        if viz_type == "Scatter plot interactif":
            st.markdown("### Scatter Plot Interactif")
            
            #Sélection des variables
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox(
                    "Variable X :",
                    X.select_dtypes(include=[np.number]).columns.tolist(),
                    index=0
                )
            
            with col2:
                y_var = st.selectbox(
                    "Variable Y :",
                    X.select_dtypes(include=[np.number]).columns.tolist(),
                    index=1
                )
            
            #Créer le scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            #Colorer par classe d'anomalie
            scatter = ax.scatter(X[x_var], X[y_var], 
                                c=y['Machine failure'], 
                                cmap='coolwarm', 
                                alpha=0.6, 
                                s=20)
            
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f"{x_var} vs {y_var}")
            ax.grid(True, alpha=0.3)
            
            #Ajouter une légende
            legend1 = ax.legend(*scatter.legend_elements(),
                               title="Panne", loc="upper right")
            ax.add_artist(legend1)
            
            st.pyplot(fig)
        
        elif viz_type == "Distribution conditionnelle":
            st.markdown("### Distribution Conditionnelle par Statut de Panne")
            
            selected_var = st.selectbox(
                "Sélectionnez une variable :",
                ['Air temperature', 'Process temperature', 
                 'Rotational speed', 'Torque', 'Tool wear']
            )
            
            #Créer un DataFrame combiné
            data_combined = X.copy()
            data_combined['Machine failure'] = y['Machine failure']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            #KDE pour chaque classe
            sns.kdeplot(data=data_combined[data_combined['Machine failure'] == 0], 
                       x=selected_var, label='Normal', fill=True, ax=ax, color='green')
            sns.kdeplot(data=data_combined[data_combined['Machine failure'] == 1], 
                       x=selected_var, label='Panne', fill=True, ax=ax, color='red')
            
            ax.set_title(f"Distribution de {selected_var} par statut de panne")
            ax.set_xlabel(selected_var)
            ax.set_ylabel("Densité")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        elif viz_type == "Analyse de clusters":
            st.info("""
            **Note :** Cette visualisation nécessite que les modèles aient été entraînés
            et que l'analyse de clustering ait été effectuée.
            """)
            
            #Vérifier si les fichiers de clustering existent
            cluster_files = [
                "latent_space_pca_analysis.png",
                "cluster_error_distribution.png"
            ]
            
            files_exist = [f for f in cluster_files if os.path.exists(f)]
            
            if files_exist:
                st.markdown("### 📊 Analyse de Clustering")
                
                for cluster_file in files_exist:
                    st.image(cluster_file, 
                            caption=cluster_file.replace('_', ' ').replace('.png', ''),
                            width='stretch')
            else:
                st.warning("""
                Les fichiers de clustering ne sont pas disponibles.
                Pour générer ces visualisations, exécutez le script d'analyse de clustering.
                """)
        
        elif viz_type == "Erreurs de reconstruction":
            st.info("""
            **Note :** Cette visualisation nécessite que les autoencodeurs aient été entraînés.
            """)
            
            #Vérifier si le fichier d'erreurs existe
            if os.path.exists("reconstruction_errors_distribution.png"):
                st.markdown("### 📈 Distribution des Erreurs de Reconstruction")
                
                st.image("reconstruction_errors_distribution.png",
                        caption="Distribution des erreurs MSE par classe",
                        width='stretch')
            else:
                st.warning("""
                Le fichier de distribution des erreurs n'est pas disponible.
                Pour générer cette visualisation, exécutez le script d'évaluation des modèles.
                """)
    
    else:
        st.error("Les données ne sont pas disponibles. Veuillez d'abord exécuter le script d'EDA.")

#FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 20px;">
    <p><strong>Dashboard de Maintenance Prédictive</strong></p>
    <p>Projet ML - Autoencodeurs pour la Détection d'Anomalies</p>
    <p>Dataset: AI4I 2020 Predictive Maintenance</p>
</div>
""", unsafe_allow_html=True)