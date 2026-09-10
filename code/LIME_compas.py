"""
Analyse de l'explicabilité locale avec LIME sur le jeu de données COMPAS.
Modèle boîte noire étudié : RandomForest.
"""

# =========================================== #
# 1. IMPORTS ET PRÉPARATION DE LA BOÎTE NOIRE #
# =========================================== #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Pour de très jolis graphiques
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime.lime_tabular

# Chargement de COMPAS (Filtres ProPublica)
url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
df = pd.read_csv(url)
df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)]
df = df[df.is_recid != -1]
df = df[df.c_charge_degree != "O"]
df = df[df.score_text != 'N/A']
''' 
- filtre : days_b_screening_arrest représente le nombre de jours entre l'arrestation et l'évaluation par le système COMPAS, si elle est trop grande
    le score COMPAS ne correspond pas à l'arrestation actuelle. On se restreint à une fenetre de +- 30 jours.
- df.is_recid != -1 : la variable is_recid est la variable cible, ici -1, correspond à si l'information n'a pas pu être trouvée. On exclut les données inconnues.
- df.c_charge_degree != "O" : correpond à des infractions au code de la route, on est dans une étude sur la recidive criminelle donc on se concentre juste sur les crimes et delits
- df.score_text != 'N/A' : On enlève les lignes NaN.
'''


features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'race']
X = df[features]

# Encodage de l'ethnie : On va se concentrer sur African-American vs Caucasian 
# pour que ce soit très "voyant" (ce sont les deux groupes majoritaires)
X = X[X['race'].isin(['African-American', 'Caucasian'])]
y = df.loc[X.index, 'two_year_recid']

X['race'] = X['race'].map({'Caucasian': 0, 'African-American': 1})

# Séparation Train/Test et conversion stricte en Numpy (.values) pour la stabilité
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_np, X_test_np = X_train.values, X_test.values

# Entraînement du modèle "Boîte Noire" (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_np, y_train.values)
print("Modèle Random Forest) entraîné avec succès.")
''' 
- n_estimators = 100 : Taille de la forêt, taille standard : le modèle est suffisamment grand pour capter des relations entre les variables
- random_state = 42 : fixe la graine aléatoire pour la construction des arbres, sans cela, les prédictions changeraient à chaque exécution
- rf_model.fit : phase d'apprentissage, le modèle étudie les profils et les compare aux vraies réponses
'''


# =============================================== #
# 2.PARAMÉTAGE DE LIME ET CRÉATION D'EXPLICATIONS #
# =============================================== #

'''
LIME cherche à expliquer une prédiction spécifique en créant un "voisinage" autour de l'individu étudié (en générant des données artificielles perturbées). Il entraîne ensuite un modèle interprétable (ex: Régression Ridge) sur ce voisinage pondéré par la distance.

Choix des Paramètres de l'Explainer :
* `discretize_continuous=True` : Les variables continues (ex: âge) sont découpées en catégories. C'est indispensable pour des juristes ou médecins qui préfèrent lire "Âge < 25 ans" plutôt que "Âge = 22.4".
* `discretizer='quartile'` : LIME découpe les données en 4 parts égales selon la distribution de la population.
* `sample_around_instance=True` : Assure que le voisinage est généré autour du point spécifique et non autour de la moyenne globale.
* training_data = X_train_np sont les données sur lesquels LIME s'entraine
* mode = classification ici la cible est catégorielle (récidive ou non) donc LIME utilise la fonction d'erreur Cross-Entropy
       Si on met regression, il attend des valeurs continues
* discretize_continuous=True et discretizer='quartile' forcent LIME à découper les variables en 4 intervalles (25%, 50% et 75% de la population)
* kernel_width = None, LIME utilise un kernel pour calculer la distance entre le vrai individu et les faux profils générés.
            Plus un faux profil est proche du vrai, plus son poids est fort dans l'entraînement de la régression locale.
            Il définit la largeur de ce voisinage, ici None  donne une largeur : sqrt(nb variables) * 0.75.
            Si la largeur est trop petite, le modèle sur-réagit au moindre bruit
            Si elle est trop grande, on n'a plus une explication locale mais une approximation globale.
'''

# Initialisation de l'Explainer avec les paramètres choisis
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=features,
    class_names=['Pas de Récidive', 'Récidive'],
    mode='classification',
    discretize_continuous=True, 
    discretizer='quartile',     
    kernel_width=None, 
    random_state=42    
)

# 1. Sélection d'un profil "à risque"
idx_cible = np.where(rf_model.predict(X_test_np) == 1)[0][0]

# 2. Génération de l'explication locale
exp_base = lime_explainer.explain_instance(
    data_row=X_test_np[idx_cible], 
    predict_fn=rf_model.predict_proba,
    num_features=5,
    num_samples=5000 
)
'''
- data_row est le profil de l'individu que l'on souhaite expliquer
- predict_fn = rf_model.predict_proba est la fonction qui permet à LIME d'interroger la boite noire.
                predict_proba renvoie une proba et non un 1 ou 0 strict
- num_features = 5 est le nombre de vaeiables qui seront affichés sur le graphe (il identifie les 5 variables qui ont le plus de poids)
- num_samples = 5000 est la taille du voisinage qu'on génère (5000 faux profils créés en perturbant les données)
'''

print("\nFacteurs poussant vers ou loin de la récidive :")
for feature, weight in exp_base.as_list():
    # Un poids positif pousse vers une récidive, un poids négatif pousse vers le contraire
    print(f"{feature}: {weight:.4f}")

# 3. Visualisation de l'explication (Correction du graphique LIME)
# On laisse LIME créer sa propre figure
fig = exp_base.as_pyplot_figure(label=1)

# On récupère l'axe de cette figure pour la modifier
ax = fig.gca()

# On applique notre titre et nos modifications
prob_recidive = rf_model.predict_proba(X_test_np[idx_cible].reshape(1, -1))[0][1]
ax.set_title(f"Explication LIME Locale : Facteurs poussant vers la récidive\n(Probabilité = {prob_recidive:.2f})")
ax.set_xlabel("Poids Local (Impact)")
fig.set_size_inches(10, 5) # On redimensionne la figure ici

plt.tight_layout()
plt.savefig("lime_explanation_compas.png", dpi=300) # Sauvegarde de la figure
plt.show()


# ================================================ #
# 3. ANALYSE D'INTERACTION : ETHNIE & ANTÉCÉDENTS  #
# ================================================ #

print("\nAnalyse d'Interaction : Poids de LIME pour le casier judiciaire en fonction de l'ethnicité...")

nb_individus_a_tester = 200
lime_weights_priors = []
valeurs_priors = []
valeurs_race = []

for i in range(nb_individus_a_tester):
    exp = lime_explainer.explain_instance(X_test.values[i], rf_model.predict_proba, num_features=5)
    
    weight_priors = 0
    for rule, weight in exp.as_list():
        if 'priors_count' in rule:
            weight_priors = weight
            break
            
    lime_weights_priors.append(weight_priors)
    valeurs_priors.append(X_test.iloc[i]['priors_count'])
    valeurs_race.append(X_test.iloc[i]['race'])

# Création du Graphique
plt.figure(figsize=(10, 6))

# On sépare les données pour faire deux nuages de points distincts
plt.scatter(np.array(valeurs_priors)[np.array(valeurs_race)==0], 
            np.array(lime_weights_priors)[np.array(valeurs_race)==0], 
            color='blue', label='Caucasian', alpha=0.6, s=80, edgecolors='k')

plt.scatter(np.array(valeurs_priors)[np.array(valeurs_race)==1], 
            np.array(lime_weights_priors)[np.array(valeurs_race)==1], 
            color='red', label='African-American', alpha=0.6, s=80, edgecolors='k')

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.title("Interaction Race / Casier Judiciaire (LIME)", fontsize=14)
plt.xlabel("Nombre d'antécédents (priors_count)")
plt.ylabel("Poids LIME du casier (Impact sur la récidive)")
plt.legend()
plt.grid(True, alpha=0.2)
plt.show()


# ====================== #
# 4. LES LIMITES DE LIME #
# ====================== #

N_ITERATIONS = 5
LIME_NUM_SAMPLES_WEAK = 300 
historique_poids = {feature: [] for feature in features}


for iteration in range(N_ITERATIONS):
    # Important : On ne fixe PAS de random_state ici pour voir la variance réelle
    exp_unstable = lime_explainer.explain_instance(
        data_row=X_test_np[idx_cible], 
        predict_fn=rf_model.predict_proba,
        num_features=len(features),
        num_samples=LIME_NUM_SAMPLES_WEAK
    )
    
    for feature_string, weight in exp_unstable.as_list():
        for base_feature in features:
            if base_feature in feature_string:
                historique_poids[base_feature].append(weight)

# Création du Graphique de l'Instabilité (Boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(historique_poids), palette="Set2")
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.title(f"Limite de LIME : Variance des poids pour le MÊME individu\n({N_ITERATIONS} exécutions | num_samples = {LIME_NUM_SAMPLES_WEAK})")
plt.ylabel("Poids assigné par LIME")
plt.xlabel("Variables du profil")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

