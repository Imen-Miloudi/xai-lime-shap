"""
Comprendre LIME avec le Titanic
Objectif : Expliquer comment une IA prédit la survie d'un passager.
Modèle : Random Forest (Boîte Noire).
"""

# ========================== #
# 1. PRÉPARATION DES DONNÉES #
# ========================== #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import lime.lime_tabular

# Chargement du dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url).dropna(subset=['Survived'])

# Sélection de variables "intelligibles" pour le tutoriel
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# --- Description des variables utilisées ---
# 1. 'Pclass' : Classe du passager (1=Élite, 2=Moyenne, 3=Économique). Proxy du statut socio-économique.
# 2. 'Sex'    : Genre encodé (0=Femme, 1=Homme). Variable historiquement pivot pour la survie ("Femmes et enfants d'abord").
# 3. 'Age'    : Âge en années. Variable continue que LIME va segmenter (ex: "Âge <= 22") pour faciliter l'interprétation.
# 4. 'SibSp'  : Nombre de frères, sœurs ou conjoints voyageant avec le passager (Siblings/Spouses).
# 5. 'Parch'  : Nombre de parents ou d'enfants voyageant avec le passager (Parents/Children).
# 6. 'Fare'   : Prix du billet. Reflet du prestige et de l'emplacement de la cabine sur le navire.

X = df[features].copy()
y = df['Survived']

# Encodage : LIME a besoin de nombres, mais nous devons garder en tête le sens
# Sex : 0 = Female, 1 = Male
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})
X['Age'] = X['Age'].fillna(X['Age'].median())

# Séparation Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement de la boîte noire
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train.values, y_train.values)
print(f"Modèle entraîné. Précision : {rf_model.score(X_test.values, y_test.values):.2f}")

# ==================================== #
# 2. CONFIGURATION DE L'EXPLAINER LIME #
# ==================================== #
''' 
- discretize_continuous=True : Transforme "Age=22" en "Age <= 22". C'est important 
  pour l'interprétabilité humaine.
- training_data : LIME utilise les stats du train pour générer ses faux profils.
'''
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=features,
    class_names=['Décédé', 'Survivant'],
    mode='classification',
    discretize_continuous=True, 
    random_state=42    
)

# ===================== #
# 3. ÉTUDE DE CAS LOCAL #
# ===================== #
# On choisit un passager spécifique (ex: index 0 du test)
idx_passager = 0
instance = X_test.iloc[idx_passager]

print(f"\n--- Analyse du Passager {idx_passager} ---")
print(f"Profil réel : \n{instance}")

# Génération de l'explication
exp = lime_explainer.explain_instance(
    data_row=instance.values, 
    predict_fn=rf_model.predict_proba,
    num_features=6,
    num_samples=5000 # On utilise 5000 pour une bonne précision locale
)

# Visualisation
fig = exp.as_pyplot_figure()
plt.title(f"Pourquoi l'IA prédit la survie de ce passager ?\n(Probabilité = {rf_model.predict_proba(instance.values.reshape(1,-1))[0][1]:.2f})")
plt.xlabel("Impact sur la décision (Vert = Survie | Rouge = Décès)")
plt.tight_layout()
plt.show()

# =============== #
# 4. Intéractions #
# =============== #
'''
L'INTERACTION CLASSE / SEXE :
On va regarder comment le poids de la variable 'Pclass' (Classe sociale)
change selon que le passager est un Homme ou une Femme.
Beaucoup plus flagrant avec ce jeu de données.
'''
print("\nLancement de l'analyse d'interaction globale...")
nb_individus = 150
poids_classe = []
valeurs_classe = []
valeurs_sexe = []

for i in range(nb_individus):
    # Explication pour chaque individu
    res = lime_explainer.explain_instance(X_test.values[i], rf_model.predict_proba, num_features=6, num_samples=1000)
    
    # On cherche le poids de Pclass
    w_pclass = 0
    for rule, weight in res.as_list():
        if 'Pclass' in rule:
            w_pclass = weight
            break
    
    poids_classe.append(w_pclass)
    valeurs_classe.append(X_test.iloc[i]['Pclass'])
    valeurs_sexe.append(X_test.iloc[i]['Sex'])

# Graphique d'interaction
plt.figure(figsize=(10, 6))
# 0 = Femme (Bleu), 1 = Homme (Rouge)
sns.scatterplot(x=valeurs_classe, y=poids_classe, hue=valeurs_sexe, 
                palette={0: 'blue', 1: 'red'}, s=100, alpha=0.7, edgecolor='k')

plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xticks([1, 2, 3])
plt.title("Comment le Sexe influence l'impact de la Classe Sociale (LIME)")
plt.xlabel("Classe du passager (1=Riche, 3=Pauvre)")
plt.ylabel("Poids LIME de la Classe (Impact sur la survie)")
plt.legend(title="Sexe", labels=['Homme (1)', 'Femme (0)'])
plt.grid(True, alpha=0.2)
plt.show()


# ==================================== #
# 5. CARTE DE CHALEUR DES INTERACTIONS #
# ==================================== #

'''
On cherche à voir globalement quelles variables influencent le plus les poids de LIME d'autres variables.
On va générer une matrice de corrélation entre les valeurs réelles des passagers et les poids de LIME pour chaque variable, puis afficher ça sous forme de carte de chaleur.
'''

print("\nGénération de la matrice d'interaction caractéristiques/poids...")

nb_individus = len(X_test) # Plus on en prend, plus la corrélation est stable
collecte_poids = []
collecte_features = []

for i in range(nb_individus):
    # On explique l'individu i et on récupère les poids de toutes les features
    exp = lime_explainer.explain_instance(X_test.values[i], rf_model.predict_proba, num_features=6, num_samples=1000)
    
    # On crée un dictionnaire temporaire pour les poids de cet individu
    dict_poids = {feat: 0.0 for feat in features}
    for rule, weight in exp.as_list():
        for f in features:
            if f in rule:
                dict_poids[f] = weight
    
    collecte_poids.append(dict_poids)
    collecte_features.append(X_test.iloc[i].to_dict())

df_poids = pd.DataFrame(collecte_poids).add_suffix('_weight')
df_valeurs = pd.DataFrame(collecte_features)

# Calcul de la corrélation croisée entre les Valeurs et les Poids
# On ne garde que la corrélation entre les colonnes de df_valeurs et df_poids
corr_matrix = pd.concat([df_valeurs, df_poids], axis=1).corr()
final_heatmap_data = corr_matrix.loc[df_valeurs.columns, df_poids.columns]

# Affichage de la Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(final_heatmap_data, annot=True, cmap='coolwarm', center=0, fmt=".2f")

plt.title("Carte de chaleur des Interactions : Quelle variable influence quel poids ?")
plt.xlabel("Poids calculés par LIME")
plt.ylabel("Valeurs réelles des passagers")
plt.tight_layout()
plt.savefig("lime_heatmap_titanic.png", dpi=300)
plt.show()
