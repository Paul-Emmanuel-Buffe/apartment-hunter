import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

modele = joblib.load(BASE_DIR / "model" / "gradient_boosting_model.joblib")
encodage_zipcode = joblib.load(BASE_DIR / "model" / "zipcode_encoding.joblib")
zones = json.loads((BASE_DIR / "data" / "zipcodes.json").read_text(encoding="utf-8"))

VARIABLES = list(modele.feature_names_in_)
assert {str(z) for z in encodage_zipcode} == set(zones), "Les codes postaux du modèle et de zipcodes.json diffèrent."

# Performances mesurées sur le jeu de test du notebook 2_modeling_and_evaluation
infos = {"modele": "HistGradientBoosting (Gradient Boosting optimisé)",
         "performances_test": {"R2_log": 0.9085, "MAE_dollars": 62972, "erreur_mediane_pct": 8.1}}
ERREUR_MEDIANE = infos["performances_test"]["erreur_mediane_pct"]

ANNEE_REFERENCE = 2015
SQFT_PAR_M2 = 10.7639
CODES_POSTAUX = sorted(zones)
MOIS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

SECTIONS = [
    {
        "titre": "Où se trouve le bien ?",
        "champs": [
            {"nom": "zipcode", "label": "Code postal", "type": "select", "defaut": "98118",
             "options": [(z, z) for z in CODES_POSTAUX],
             "aide": "Les 70 codes postaux du comté de King (région de Seattle)."},
        ],
    },
    {
        "titre": "Surfaces",
        "champs": [
            {"nom": "living_m2", "label": "Surface habitable (m²)", "type": "number", "min": 30, "max": 930, "step": 1, "defaut": 150},
            {"nom": "above_m2", "label": "Surface hors sous-sol (m²)", "type": "number", "min": 20, "max": 930, "step": 1, "optionnel": True,
             "aide": "Laissez vide pour la déduire : surface habitable moins sous-sol."},
            {"nom": "basement_m2", "label": "Surface du sous-sol (m²)", "type": "number", "min": 0, "max": 900, "step": 1, "defaut": 0,
             "aide": "0 si le bien n'a pas de sous-sol."},
            {"nom": "lot_m2", "label": "Surface du terrain (m²)", "type": "number", "min": 40, "max": 70000, "step": 1, "defaut": 600},
        ],
    },
    {
        "titre": "Pièces et étages",
        "champs": [
            {"nom": "bedrooms", "label": "Chambres", "type": "number", "min": 0, "max": 10, "step": 1, "defaut": 3,
             "aide": "0 = studio."},
            {"nom": "bathrooms", "label": "Salles d'eau", "type": "number", "min": 0.5, "max": 8, "step": 0.25, "defaut": 2,
             "aide": "1 = salle de bain complète, 0,5 = wc et lavabo seuls, 0,75 = avec douche sans baignoire."},
            {"nom": "floors", "label": "Étages", "type": "number", "min": 1, "max": 3.5, "step": 0.5, "defaut": 1},
        ],
    },
    {
        "titre": "Âge et vente",
        "champs": [
            {"nom": "yr_built", "label": "Année de construction", "type": "number", "min": 1900, "max": 2015, "step": 1, "defaut": 1975,
             "aide": "Sert à calculer l'ancienneté du bien (2015 = année de référence des données)."},
            {"nom": "yr_renovated", "label": "Année de la dernière rénovation", "type": "number", "min": 1900, "max": 2015, "step": 1, "optionnel": True,
             "aide": "Laissez vide si le bien n'a jamais été rénové. Sert à calculer l'indicateur de rénovation et l'âge effectif."},
            {"nom": "mois_vente", "label": "Mois de vente", "type": "select", "defaut": "5",
             "options": [(i + 1, nom) for i, nom in enumerate(MOIS)]},
        ],
    },
    {
        "titre": "Qualité et environnement",
        "champs": [
            {"nom": "grade", "label": "Qualité de construction (3 à 13)", "type": "number", "min": 3, "max": 13, "step": 1, "defaut": 7,
             "aide": "7 = qualité moyenne, 11 à 13 = haut de gamme."},
            {"nom": "condition", "label": "État général (1 à 5)", "type": "number", "min": 1, "max": 5, "step": 1, "defaut": 3},
            {"nom": "view", "label": "Qualité de la vue", "type": "select", "defaut": "0",
             "options": [(0, "0 - Aucune vue particulière"), (1, "1 - Passable"), (2, "2 - Moyenne"), (3, "3 - Bonne"), (4, "4 - Excellente")],
             "aide": "Indice de 0 à 4 utilisé dans les données d'origine."},
            {"nom": "waterfront", "label": "Vue sur l'eau (front de mer ou de lac)", "type": "checkbox", "defaut": False},
        ],
    },
    {
        "titre": "Options avancées",
        "avance": True,
        "champs": [
            {"nom": "lat", "label": "Latitude", "type": "number", "min": 47.1, "max": 47.8, "step": "any", "optionnel": True,
             "aide": "Par défaut : centre du code postal."},
            {"nom": "long", "label": "Longitude", "type": "number", "min": -122.6, "max": -121.2, "step": "any", "optionnel": True},
            {"nom": "living15_m2", "label": "Surface habitable moyenne des 15 voisins (m²)", "type": "number", "min": 30, "max": 930, "step": 1, "optionnel": True,
             "aide": "Par défaut : médiane du code postal."},
            {"nom": "lot15_m2", "label": "Surface de terrain moyenne des 15 voisins (m²)", "type": "number", "min": 40, "max": 70000, "step": 1, "optionnel": True},
        ],
    },
]
TOUS_LES_CHAMPS = [champ for section in SECTIONS for champ in section["champs"]]


def lire_champs(donnees):
    valeurs, erreurs = {}, {}
    for champ in TOUS_LES_CHAMPS:
        nom, brut = champ["nom"], donnees.get(champ["nom"])
        if champ["type"] == "checkbox":
            valeurs[nom] = str(brut).strip().lower() in ("on", "true", "1")
            continue
        texte = "" if brut is None else str(brut).strip().replace(",", ".")
        if texte == "":
            if champ.get("optionnel"):
                valeurs[nom] = None
            else:
                erreurs[nom] = "Ce champ est obligatoire."
            continue
        if champ["type"] == "select":
            if texte in {str(valeur) for valeur, _ in champ["options"]}:
                valeurs[nom] = int(texte)
            else:
                erreurs[nom] = "Valeur non reconnue."
            continue
        try:
            nombre = float(texte)
        except ValueError:
            erreurs[nom] = "Saisissez un nombre."
            continue
        if not champ["min"] <= nombre <= champ["max"]:
            erreurs[nom] = f"Valeur attendue entre {champ['min']} et {champ['max']}."
            continue
        valeurs[nom] = nombre

    if "basement_m2" not in erreurs and "living_m2" not in erreurs:
        if valeurs["basement_m2"] >= valeurs["living_m2"]:
            erreurs["basement_m2"] = "Le sous-sol doit être plus petit que la surface habitable."
        elif valeurs.get("above_m2") is not None and "above_m2" not in erreurs:
            if abs(valeurs["above_m2"] + valeurs["basement_m2"] - valeurs["living_m2"]) > 1:
                erreurs["above_m2"] = "La surface hors sous-sol plus le sous-sol doit égaler la surface habitable."
    if valeurs.get("yr_renovated") is not None and "yr_built" not in erreurs and valeurs["yr_renovated"] < valeurs["yr_built"]:
        erreurs["yr_renovated"] = "La rénovation ne peut pas précéder la construction."
    return valeurs, erreurs


def construire_variables(v):
    zone = zones[str(v["zipcode"])]
    lat = zone["lat"] if v["lat"] is None else v["lat"]
    long = zone["long"] if v["long"] is None else v["long"]
    living15 = zone["living15_m2"] if v["living15_m2"] is None else v["living15_m2"]
    lot15 = zone["lot15_m2"] if v["lot15_m2"] is None else v["lot15_m2"]

    sqft_living = round(v["living_m2"] * SQFT_PAR_M2)
    sqft_basement = round(v["basement_m2"] * SQFT_PAR_M2)
    sqft_above = sqft_living - sqft_basement if v["above_m2"] is None else round(v["above_m2"] * SQFT_PAR_M2)
    anciennete = ANNEE_REFERENCE - int(v["yr_built"])
    renove = v["yr_renovated"] is not None
    ligne = {
        "bedrooms": int(v["bedrooms"]),
        "bathrooms": v["bathrooms"],
        "sqft_living": sqft_living,
        "sqft_lot": round(v["lot_m2"] * SQFT_PAR_M2),
        "floors": v["floors"],
        "waterfront": int(v["waterfront"]),
        "view": int(v["view"]),
        "condition": int(v["condition"]),
        "grade": int(v["grade"]),
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "lat": lat,
        "long": long,
        "sqft_living15": round(living15 * SQFT_PAR_M2),
        "sqft_lot15": round(lot15 * SQFT_PAR_M2),
        "anciennete": anciennete,
        "est_renove": int(renove),
        "age_effectif": ANNEE_REFERENCE - int(v["yr_renovated"]) if renove else anciennete,
        "mois_vente": int(v["mois_vente"]),
        "zipcode_encoded": encodage_zipcode[v["zipcode"]],
    }
    return pd.DataFrame([ligne])[VARIABLES]


def estimer(valeurs):
    variables = construire_variables(valeurs)
    prix = float(np.exp(modele.predict(variables)[0]))
    marge = ERREUR_MEDIANE / 100
    return {
        "prix": prix,
        "bas": prix * (1 - marge),
        "haut": prix * (1 + marge),
        "prix_m2": prix / valeurs["living_m2"],
        "variables": variables.iloc[0].to_dict(),
    }


@app.template_filter("dollars")
def filtre_dollars(valeur):
    arrondi = int(round(valeur / 1000.0) * 1000)
    return f"{arrondi:,}".replace(",", " ") + " $"


@app.template_filter("dollars_exact")
def filtre_dollars_exact(valeur):
    return f"{int(round(valeur)):,}".replace(",", " ") + " $"


@app.template_filter("nb")
def filtre_nombre(valeur):
    if isinstance(valeur, float) and valeur.is_integer():
        return int(valeur)
    if isinstance(valeur, float):
        return round(valeur, 4)
    return valeur


@app.route("/", methods=["GET", "POST"])
def accueil():
    saisie = {c["nom"]: c.get("defaut", "") for c in TOUS_LES_CHAMPS}
    erreurs, resultat = {}, None
    if request.method == "POST":
        saisie = {c["nom"]: request.form.get(c["nom"], "") for c in TOUS_LES_CHAMPS}
        valeurs, erreurs = lire_champs(request.form)
        if not erreurs:
            resultat = estimer(valeurs)
    return render_template("index.html", sections=SECTIONS, saisie=saisie, erreurs=erreurs, resultat=resultat,
                           zones=zones, infos=infos, erreur_mediane=ERREUR_MEDIANE)


@app.post("/api/estimation")
def api_estimation():
    donnees = request.get_json(silent=True)
    if not isinstance(donnees, dict):
        return jsonify({"erreurs": {"corps": "Envoyez un objet JSON."}}), 400
    valeurs, erreurs = lire_champs(donnees)
    if erreurs:
        return jsonify({"erreurs": erreurs}), 400
    r = estimer(valeurs)
    return jsonify({
        "prix_estime": round(r["prix"]),
        "fourchette": [round(r["bas"]), round(r["haut"])],
        "prix_m2": round(r["prix_m2"]),
        "devise": "USD",
        "erreur_mediane_pct": ERREUR_MEDIANE,
        "variables_modele": r["variables"],
    })


@app.get("/sante")
def sante():
    return jsonify({"statut": "ok"})


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5001)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
