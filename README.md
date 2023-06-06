# OC Projet 7 - Implémentez un modèle de scoring
**Prêt à dépenser** est une société financière qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées. Cet algorithme implémentera un Dashboard interactif à travers d'une API de prédiction pour expliquer de façon la plus transparente possible les décisions d’octroi de crédit.

Liens vers les répertoires Github de ce projet:
* Les notebooks du projet : 
  * https://github.com/Rimdata/Implementez_un_modele_de_scoring
* L'API de prédiction :
  * https://github.com/Rimdata/p7_predictapi
  * https://predictapi.herokuapp.com/
* Le tableau de bord : 
  * https://github.com/Rimdata/P7_creditapp
  * https://credit-app.herokuapp.com/

# Gestionnaire de crédit - Les notebooks de modélisation

Ce répertoire contient les notebooks de modélisation pour le projet de scoring de crédit de l'entreprise "Prêt à dépenser".
Voici une description des différents fichiers et dossiers présents :

* **BAHROUN_Rim_1_notebook_042023.ipynb** : Agrégation des données
  * Agrégation des données afin d'obtenir un seul jeu de données.
  * Création de quelques variables intéressantes pour le projet.
  * Correction des valeurs abérrantes.
  * Teste d'un modèle basé sur lightgbm sur tous les données agrégées.
  * Un jeu de données d’entrainement de 797 variables et 356 251 clients.
  * Ce notebook est inspirée du kernel kaggle https://www.kaggle.com/code/tunguz/xgb-simple-features 
 
* **BAHROUN_Rim_2_notebook_042023.ipynb** : Sélection des variables pertinentes -1/2
  * Suppression des variables collinéaires.
  * Suppression des variables avec un pourcentage de valeurs manquantes supérieur à un seuil.
  * Suppression dess variables les moins pertinentes en utilisant les importances des variables à partir d'un modèle.
  * Suppression des variables avec faible variance.
  * Un jeu de données d’entrainement de 99 variables et 356 251 clients.

* **BAHROUN_Rim_3_notebook_042023.ipynb** : Sélection des variables pertinentes -2/2
  * Sélection des variables par la méthode SelectKBest
  * Sélection des variables par la méthode RFE
  * Un jeu de données d’entrainement de 35 variables et 356 251 clients.

* **BAHROUN_Rim_4_notebook_042023.ipynb** : Modélisation
  * Résolution du problème de désiquilibre entre les classes.
  * Implémentation des modèles de machine learning pour l'apprentissage supervisé 
  * Recherche des meilleurs hyperparamètres par gridsearchCV
  * Comparaison des performances des modèles
  * Sélection du meilleur modèle
  * Interprétation globale et locale du modèle

* **BAHROUN_Rim_5_notebook_042023.ipynb** : Préparation du dashboard
  * Rappel du modèle sélectionné et ses performances.
  * Sauvegarde du modèle.
  * Préparation du code pour le Dashboard

* **BAHROUN_Rim_6_notebook_042023.ipynb** : Data Drift
  * Analyse du data drift avec la librairie Evidently
