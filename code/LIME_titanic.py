"""
XAI pour le Diagnostic Médical : Analyse du Breast Cancer avec LIME.
Modèle : XGBoost.
Objectif : Identifier les facteurs de malignité et évaluer la robustesse locale.
"""

# ========================== #
# 1. PRÉPARATION DES DONNÉES #
# ========================== #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import lime.lime_tabular

def setup_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target # 0 = Maligne, 1 = Bénigne
    
    # --- Description des variables clés ---
    # 'mean radius' : Taille moyenne des noyaux cellulaires.
    # 'mean texture' : Écart-type des niveaux de gris (densité).
    # 'worst area' : La plus grande surface mesurée.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, data.feature_names, data.target_names

# Entraînement de la boîte noire
X_train, X_test, y_train, y_test, feature_names, class_names = setup_cancer_data()
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train.values, y_train)

# ==================================== #
# 2. CONFIGURATION DE L'EXPLAINER LIME #
# ==================================== #
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=class_names, # malignant (0), benign (1)
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# ===================== #
# 3. ÉTUDE DE CAS LOCAL #
# ===================== #
# Sélection d'une patiente diagnostiquée Maligne (Classe 0)
idx_maligne = np.where((y_test == 0) & (xgb_model.predict(X_test.values) == 0))[0][0]
instance = X_test.iloc[idx_maligne]

exp = lime_explainer.explain_instance(
    data_row=instance.values, 
    predict_fn=xgb_model.predict_proba,
    num_features=8,
    labels=(0,)
)

fig = exp.as_pyplot_figure(label=0)
plt.title(f"Diagnostic LIME : Pourquoi ce cas est malin ?\n(Confiance XGBoost : {xgb_model.predict_proba(instance.values.reshape(1,-1))[0][0]:.2f})")
plt.tight_layout()
plt.show()

# ==================================== #
# 4. CARTE DE CHALEUR DES INTERACTIONS #
# ==================================== #
print("\nGénération de la Heatmap des interactions...")
nb_individus = len(X_test)
collecte_poids = []
for i in range(nb_individus):
    res = lime_explainer.explain_instance(
        X_test.values[i], 
        xgb_model.predict_proba, 
        num_features=len(feature_names),
        labels=(0,)
    )
    dict_poids = {feat: 0.0 for feat in feature_names}
    for rule, weight in res.as_list(label=0):
        for f in feature_names:
            if f in rule: dict_poids[f] = weight
    collecte_poids.append(dict_poids)

df_poids = pd.DataFrame(collecte_poids).add_suffix('_weight')
corr_matrix = pd.concat([X_test.reset_index(drop=True), df_poids], axis=1).corr()
final_heatmap_data = corr_matrix.loc[X_test.columns[:10], df_poids.columns[:10]] # On limite aux 10 premières variables

plt.figure(figsize=(12, 10))
sns.heatmap(final_heatmap_data, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Matrice de Corrélation Valeurs-Poids (Diagnostic Cancer)")
plt.tight_layout()
plt.savefig("lime_heatmap_cancer.png", dpi=300)
plt.show()

# ========================================== #
# 5. INTERACTIONS : CONCAVE POINTS vs RADIUS #
# ========================================== #
print("\nAnalyse d'interaction basée sur la Heatmap...")

nb_individus = len(X_test)
poids_concave = []
val_concave = []
val_radius = []

for i in range(nb_individus):
    # On force LIME à expliquer la classe 'Malignant' (label 0)
    res = lime_explainer.explain_instance(
        X_test.values[i], 
        xgb_model.predict_proba, 
        num_features=len(feature_names),
        labels=(0,)
    )
    
    # On extrait le poids de la variable 'mean concave points'
    w_concave = 0
    for rule, weight in res.as_list(label=0):
        if 'mean concave points' in rule:
            w_concave = weight
            break
            
    poids_concave.append(w_concave)
    val_concave.append(X_test.iloc[i]['mean concave points'])
    val_radius.append(X_test.iloc[i]['mean radius'])

# Création du Graphique
plt.figure(figsize=(10, 6))
scatter = plt.scatter(val_concave, poids_concave, c=val_radius, 
                     cmap='YlOrRd', s=100, alpha=0.7, edgecolors='k')

cbar = plt.colorbar(scatter)
cbar.set_label("Rayon moyen (mean radius) en mm")
plt.axhline(0, color='black', linestyle='--', alpha=0.5)

plt.title("Interaction LIME : Impact des points concaves modulé par le rayon", fontsize=13)
plt.xlabel("Valeur réelle de 'mean concave points'")
plt.ylabel("Poids LIME (Influence sur le diagnostic Malin)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("interaction_concave_radius.png", dpi=300)
plt.show()

# =================================== #
# 6. LIMITES : LE VOISINAGE (KERNEL)  #
# =================================== #
kernel_widths = [1.5, 4.0, 15.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, kw in enumerate(kernel_widths):
    temp_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values, feature_names=feature_names, class_names=class_names,
        mode='classification', discretize_continuous=True, random_state=42, kernel_width=kw
    )
    res = temp_explainer.explain_instance(instance.values, xgb_model.predict_proba, num_features=8, labels=(0,))
    
    weights = [x[1] for x in res.as_list(label=0)]
    feats = [x[0] for x in res.as_list(label=0)]
    axes[i].barh(feats, weights, color=['#d62728' if w > 0 else '#2ca02c' for w in weights])
    axes[i].set_title(f"Kernel Width = {kw}")
    axes[i].invert_yaxis()

plt.suptitle("Instabilité de LIME : Impact du voisinage sur le diagnostic médical", fontsize=16)
plt.tight_layout()
plt.savefig("lime_instabilite_cancer.png", dpi=300)
plt.show()
