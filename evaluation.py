import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    fbeta_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def find_optimal_threshold(y_true, scores, metric='f2', min_recall=0.85):
    #trouver le seuil optimal basé sur une métrique spécifique.

    if metric == 'f1':
        beta = 1
    elif metric == 'f2':  #Plus sensible au rappel
        beta = 2
    elif metric == 'f0.5':  #Plus sensible à la précision
        beta = 0.5
    else:
        raise ValueError("metric doit être 'f1', 'f2' ou 'f0.5'")
    
    # Trier les scores
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_y_true = y_true[sorted_indices]
    
    best_threshold = sorted_scores[0] - 0.1  #Initialisation
    best_fbeta = 0
    best_y_pred = None
    
    #Chercher le meilleur seuil
    for i in range(len(sorted_scores)):
        threshold = sorted_scores[i]
        y_pred = (scores > threshold).astype(int)
        
        #Calculer recall
        recall_val = recall_score(y_true, y_pred, zero_division=0)
        
        #Si recall < minimum requis, continuer
        if recall_val < min_recall:
            continue
        
        # Calculer F-beta
        fbeta_val = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        
        if fbeta_val > best_fbeta:
            best_fbeta = fbeta_val
            best_threshold = threshold
            best_y_pred = y_pred
    
    return best_threshold, best_fbeta, best_y_pred


def evaluate_model_complete(y_true, y_pred, scores, model_name="Modèle"):
    #evaluation complète d'un modèle de détection d'anomalies.

    #Vérifications
    if len(y_true) != len(y_pred):
        raise ValueError(f"Dimensions incompatibles: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    #Calcul des métriques de base
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    
    #AUC-ROC
    auc_roc = roc_auc_score(y_true, scores)
    
    #AUC-PR
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)
    
    #Rapport détaillé
    print(f"\n{'='*60}")
    print(f"ÉVALUATION COMPLÈTE - {model_name}")
    print(f"{'='*60}")
    
    print(f"\nMÉTRIQUES DE BASE:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  F2-Score:  {f2:.4f} (favorise recall)")
    print(f"  F0.5-Score: {f05:.4f} (favorise precision)")
    print(f"  AUC-ROC:   {auc_roc:.4f}")
    print(f"  AUC-PR:    {auc_pr:.4f}")
    
    print(f"\nMATRICE DE CONFUSION:")
    print(f"  Vrais Négatifs (TN): {tn:5d}")
    print(f"  Faux Positifs (FP):  {fp:5d}")
    print(f"  Faux Négatifs (FN):  {fn:5d}")
    print(f"  Vrais Positifs (TP): {tp:5d}")
    print(f"  Taux de détection:   {tp/(tp+fn):.2%}")
    print(f"  Taux de fausses alarmes: {fp/(fp+tn):.2%}")
    
    # Rapport de classification sklearn
    print(f"\nRAPPORT DE CLASSIFICATION:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalie"], digits=4))
    
    # Retourner toutes les métriques dans un dictionnaire
    metrics_dict = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'F2-Score': f2,
        'F0.5-Score': f05,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Detection_Rate': tp/(tp+fn) if (tp+fn) > 0 else 0,
        'False_Alarm_Rate': fp/(fp+tn) if (fp+tn) > 0 else 0
    }
    
    return metrics_dict


def plot_confusion_matrices(y_true_list, y_pred_list, model_names, save_path=None):
    #afficher plusieurs matrices de confusion côte à côte.

    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for i, (y_true, y_pred, name) in enumerate(zip(y_true_list, y_pred_list, model_names)):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Anomalie"]
        )
        
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False, values_format='d')
        axes[i].set_title(f"{name}\nTP={cm[1,1]}, FP={cm[0,1]}\nFN={cm[1,0]}, TN={cm[0,0]}")
        
        # Ajouter les pourcentages
        for (j, k), val in np.ndenumerate(cm):
            total = cm.sum()
            percentage = val / total * 100
            axes[i].text(k, j, f'{val}\n({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_roc_pr_comparison(results_dict, y_true_dict, scores_dict, save_dir='.'):
    #Tracer la courbe ROC pour comparaison.
    plt.figure(figsize=(12, 5))

    for name in results_dict.keys():
        if name in scores_dict and name in y_true_dict:
            y_true = y_true_dict[name]
            scores = scores_dict[name]
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc_roc = results_dict[name]['AUC-ROC']
            
            linestyle = '-' if 'AE' in name or 'autoencoder' in name.lower() else '--'
            plt.plot(fpr, tpr, linestyle=linestyle, linewidth=2, 
                    label=f"{name} (AUC={auc_roc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Aléatoire (AUC=0.5)')
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Comparaison des courbes ROC – Tous les modèles", fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_metrics_comparison_bar(results_df, save_path=None):
    #Afficher un bar chart comparatif des métriques principales.

    metrics_to_plot = ['Recall', 'F2-Score', 'Precision', 'AUC-ROC', 'AUC-PR']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        sorted_df = results_df.sort_values(metric, ascending=True)
        
        #Couleurs différentes pour autoencodeurs vs classiques
        colors = ['steelblue' if 'AE' in name or 'autoencoder' in name.lower() 
                 else 'darkorange' for name in sorted_df['Model']]
        
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['Model'], fontsize=9)
        ax.set_xlabel(metric, fontsize=10)
        ax.set_title(f'{metric} - Comparaison', fontsize=11, fontweight='bold')
        ax.set_xlim([0, 1.05])
        
        #Ajouter les valeurs
        for i, (bar, val) in enumerate(zip(bars, sorted_df[metric])):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)
    
    #Dernier subplot: métriques composites
    ax = axes[len(metrics_to_plot)]
    models = results_df['Model']
    recall_vals = results_df['Recall'].values
    precision_vals = results_df['Precision'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, recall_vals, width, label='Recall', color='lightcoral', alpha=0.8)
    ax.bar(x + width/2, precision_vals, width, label='Precision', color='lightblue', alpha=0.8)
    
    ax.set_xlabel('Modèles')
    ax.set_ylabel('Score')
    ax.set_title('Recall vs Precision', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim([0, 1.05])
    
    #Cacher les subplots vides
    for i in range(len(metrics_to_plot) + 1, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_detailed_report(results_df, save_path='model_comparison_detailed.csv'):
    #génère un rapport détaillé des performances.

    # Ajouter des métriques supplémentaires
    results_df['TPR'] = results_df['TP'] / (results_df['TP'] + results_df['FN'])
    results_df['FPR'] = results_df['FP'] / (results_df['FP'] + results_df['TN'])
    results_df['FNR'] = results_df['FN'] / (results_df['TP'] + results_df['FN'])
    results_df['TNR'] = results_df['TN'] / (results_df['TN'] + results_df['FP'])
    
    # Calculer le score composite (moyenne pondérée)
    weights = {
        'Recall': 0.3,      # Important pour la détection d'anomalies
        'F2-Score': 0.3,    # Favorise le recall
        'AUC-PR': 0.2,      # Bon pour déséquilibre de classes
        'Precision': 0.1,   # Moins critique pour anomalies
        'AUC-ROC': 0.1      # Standard
    }
    
    results_df['Composite_Score'] = 0
    for metric, weight in weights.items():
        if metric in results_df.columns:
            results_df['Composite_Score'] += results_df[metric] * weight
    
    #Trier par score composite
    results_df = results_df.sort_values('Composite_Score', ascending=False)
    
    #Formater les pourcentages
    percent_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'F2-Score', 
                   'F0.5-Score', 'AUC-ROC', 'AUC-PR', 'TPR', 'FPR', 'FNR', 'TNR',
                   'Detection_Rate', 'False_Alarm_Rate', 'Composite_Score']
    
    for col in percent_cols:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: f"{x:.3f}")
    
    #Sauvegarder
    results_df.to_csv(save_path, index=False)
    
    print(f"\nRAPPORT DÉTAILLÉ GÉNÉRÉ : {save_path}")
    print("\nClassement des modèles (score composite):")
    for i, row in results_df.iterrows():
        print(f"{i+1:2d}. {row['Model']:20s} - Score: {row['Composite_Score']}")
    
    return results_df


def plot_reconstruction_error_distribution(mse_scores, y_true, threshold, model_name, save_path=None):

    #Visualiser la distribution des erreurs de reconstruction.

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Histogramme des erreurs
    ax = axes[0]
    ax.hist(mse_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, 
               label=f'Seuil = {threshold:.4f}')
    ax.set_xlabel('Erreur de reconstruction (MSE)')
    ax.set_ylabel('Fréquence')
    ax.set_title(f'Distribution des erreurs - {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Boxplot par classe
    ax = axes[1]
    normal_scores = mse_scores[y_true == 0]
    anomaly_scores = mse_scores[y_true == 1]
    
    bp = ax.boxplot([normal_scores, anomaly_scores], 
                    labels=['Normal', 'Anomalie'],
                    patch_artist=True)
    
    # Colorer les boxplots
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_ylabel('Erreur MSE')
    ax.set_title(f'Erreurs par classe - {model_name}')
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Densité par classe
    ax = axes[2]
    if len(normal_scores) > 0:
        sns.kdeplot(normal_scores, ax=ax, label='Normal', color='green', fill=True, alpha=0.3)
    if len(anomaly_scores) > 0:
        sns.kdeplot(anomaly_scores, ax=ax, label='Anomalie', color='red', fill=True, alpha=0.3)
    
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Seuil')
    ax.set_xlabel('Erreur MSE')
    ax.set_ylabel('Densité')
    ax.set_title(f'Densité des erreurs - {model_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()