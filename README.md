# Carte Interactive des Incidents – Haute-Garonne


https://github.com/user-attachments/assets/50796805-b41f-4320-a7f9-caabdbe08350



Ce projet est une application interactive développée avec **Gradio** et **Plotly**, permettant de visualiser et d'explorer les incidents d'urgence (incendies, accidents, etc.) dans le département de la Haute-Garonne, ainsi que les casernes de pompiers et leur niveau de qualification.

## Fonctionnalités principales

- **Visualisation géographique** des incidents et casernes sur une carte interactive.
- **Filtrage dynamique** des incidents par type, mois et nombre maximal à afficher.
- **Statistiques détaillées** par incident : temps d'intervention, nombre de pompiers ou véhicules sous-qualifiés, distances moyennes, etc.
- **Génération de graphiques** en temps réel avec Matplotlib.
- Interface réactive grâce à **Gradio Blocks**.

##  Structure des fichiers

- `incidents.csv` : Données des incidents (colonnes attendues : `id`, `nom`, `type`, `lat`, `lon`, `mois`, `pompiers_sous_q`, `vehicules_sous_q`, `temps`, `distance`)
- `casernes.csv` : Données des casernes (colonnes attendues : `id`, `nom`, `lat`, `lon`, `niveau_qualification`)
- `test2.py` : Code principal de l'application
- `README.md` : Ce fichier








