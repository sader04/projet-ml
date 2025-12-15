!pip install ucimlrepo
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

#configuration
FIGURES_DIR = "reports/figures"
PROCESSED_DATA_DIR = "data/processed"

#créer les dossiers si nécessaires
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

#variables quantitatives principales
QUANT_FEATURES = [
    'Air temperature',
    'Process temperature',
    'Rotational speed',
    'Torque',
    'Tool wear'
]

FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']


def load_and_prepare_data(save_processed=True):
    #charger depuis UCI
    data = fetch_ucirepo(id=601)
    X = data.data.features.copy()
    y = data.data.targets.copy()

    #ajouter Delta_T
    X['Delta_T'] = X['Process temperature'] - X['Air temperature']

    #encoder la variable 'Type' (qualité du produit)
    X_encoded = pd.get_dummies(X, columns=['Type'], prefix='Type')

    if save_processed:
        X.to_csv(os.path.join(PROCESSED_DATA_DIR, "data_raw_with_deltas.csv"), index=False)
        X_encoded.to_csv(os.path.join(PROCESSED_DATA_DIR, "data_prepared.csv"), index=False)
        y.to_csv(os.path.join(PROCESSED_DATA_DIR, "targets.csv"), index=False)

    return X, X_encoded, y


def plot_distributions(X, save=True):
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(QUANT_FEATURES, 1):
        plt.subplot(2, 3, i)
        sns.histplot(X[col], kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution de {col}')
        plt.xlabel(col)
        plt.ylabel('Fréquence')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig1_distributions.png'), dpi=150)
    plt.show()


def plot_boxplots_by_failure(X, y, save=True):
    y_failure = y['Machine failure'].astype('category')
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(QUANT_FEATURES, 1):
        plt.subplot(1, 5, i)
        sns.boxplot(
            y=X[col],
            x=y_failure,
            hue=y_failure,
            palette=['#2ca02c', '#d62728'],
            legend=False
        )
        plt.title(col)
        plt.xlabel('Machine failure')
        plt.ylabel(col)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig2_boxplots_failure.png'), dpi=150)
    plt.show()


def plot_product_quality(X, save=True):
    product_counts = X['Type'].value_counts().reindex(['L', 'M', 'H'])
    product_pct = product_counts / len(X) * 100

    plt.figure(figsize=(6, 5))
    custom_colors = ['#FF0000', '#FFFF00', '#00FF00']  
    ax = sns.barplot(
        x=product_counts.index,
        y=product_counts.values,
        hue=product_counts.index,
        palette=custom_colors,
        legend=False
    )
    plt.title('Répartition des types de produit')
    plt.xlabel('Qualité du produit')
    plt.ylabel("Nombre d'observations")

    for i, (count, pct) in enumerate(zip(product_counts, product_pct)):
        ax.text(i, count + 50, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig3_product_quality.png'), dpi=150)
    plt.show()


def plot_failure_modes(y, save=True):
    failure_counts = y[FAILURE_MODES].sum().sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(failure_counts.index, failure_counts.values, color='#D2042D')
    plt.title('Fréquence des modes de panne')
    plt.xlabel("Nombre d'occurrences")
    plt.ylabel("Type de panne")

    for bar in bars:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 int(bar.get_width()), va='center', ha='left', fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig4_failure_modes.png'), dpi=150)
    plt.show()


def plot_correlation_heatmap(X_encoded, save=True):
    numeric_cols = [col for col in X_encoded.columns if not col.startswith('Type_')]
    corr = X_encoded[numeric_cols].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        center=0,
        linewidths=0.5
    )
    plt.title('Matrice de corrélation (variables continues + Type encodé)')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig5_correlation_heatmap.png'), dpi=150)
    plt.show()


def plot_critical_pairs(X, y, save=True):
    colors = y['Machine failure'].map({0: 'green', 1: 'red'})
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    #ΔT vs Rotational speed (HDF)
    ax = axes[0, 0]
    ax.scatter(X['Delta_T'], X['Rotational speed'], c=colors, alpha=0.6) 
    ax.axhline(1380, color='black', linestyle='--', label='1380 rpm')
    ax.axvline(8.6, color='black', linestyle='--', label='ΔT = 8.6 K')
    ax.set_xlabel('ΔT = Process - Air temp [K]')
    ax.set_ylabel('Rotational speed [rpm]')
    ax.set_title('HDF: Heat Dissipation Failure')
    ax.legend()

    #Torque vs Rotational speed (PWF)
    ax = axes[0, 1]
    ax.scatter(X['Torque'], X['Rotational speed'], c=colors, alpha=0.6) s
    rpm = np.linspace(X['Rotational speed'].min(), X['Rotational speed'].max(), 100) 
    for P in [3500, 9000]:
        torque_iso = P / (rpm * 2 * np.pi / 60)
        ax.plot(torque_iso, rpm, '--', label=f'Power = {P} W')
    ax.set_xlabel('Torque [Nm]')
    ax.set_ylabel('Rotational speed [rpm]')
    ax.set_title('PWF: Power Failure')
    ax.legend()

    #Torque vs Tool wear (OSF)
    ax = axes[1, 0]
    df_plot = X.copy()
    sns.scatterplot(
        data=df_plot,
        x='Tool wear',
        y='Torque',
        hue='Type',
        palette={'L': 'blue', 'M': 'orange', 'H': 'purple'},
        alpha=0.7,
        ax=ax
    ) 
    wear = np.linspace(0.1, X['Tool wear'].max(), 100) 
    thresholds = {'L': 11000, 'M': 12000, 'H': 13000}
    for key, thresh in thresholds.items():
        torque_thresh = thresh / wear
        ax.plot(wear, torque_thresh, '--', color={'L': 'blue', 'M': 'orange', 'H': 'purple'}[key],
                label=f'Seuil {key} ({thresh})')
    ax.set_xlabel('Tool wear [min]')
    ax.set_ylabel('Torque [Nm]')
    ax.set_title('OSF: Overstrain Failure')
    ax.legend()

    #Tool wear histogram (TWF)
    ax = axes[1, 1]
    sns.histplot(
        x=X['Tool wear'],
        hue=y['Machine failure'],
        multiple='stack',
        palette=['green', 'red'],
        bins=50,
        ax=ax
    ) 
    ax.axvspan(200, 240, color='red', alpha=0.2, label='Zone TWF (200–240 min)')
    ax.set_xlabel('Tool wear [min]')
    ax.set_ylabel('Fréquence')
    ax.set_title('TWF: Tool Wear Failure')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig6_critical_pairs.png'), dpi=150)
    plt.show()


def plot_conditional_kde(X, y, save=True):
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes = axes.flatten()
    for i, col in enumerate(QUANT_FEATURES):
        ax = axes[i]
        sns.kdeplot(
            data=X,
            x=col,
            hue=y['Machine failure'],
            palette={0: 'green', 1: 'red'},
            fill=True,
            alpha=0.4,
            ax=ax
        )
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Densité')
    for j in range(len(QUANT_FEATURES), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save:
        plt.savefig(os.path.join(FIGURES_DIR, 'fig7_conditional_kde.png'), dpi=150)
    plt.show()


def run_full_eda():
    print("Chargement et préparation des données...")
    X, X_encoded, y = load_and_prepare_data()

    print("Génération des figures...")
    plot_distributions(X)
    plot_boxplots_by_failure(X, y)
    plot_product_quality(X)
    plot_failure_modes(y)
    plot_correlation_heatmap(X_encoded)
    plot_critical_pairs(X, y)
    plot_conditional_kde(X, y)

    print("EDA terminée. Données et figures sauvegardées.")


if __name__ == "__main__":
    run_full_eda()