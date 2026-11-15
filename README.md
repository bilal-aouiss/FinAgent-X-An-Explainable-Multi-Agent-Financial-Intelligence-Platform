# FinAgent-X — An Explainable Multi-Agent Financial Intelligence Platform

FinAgent-X couvre officiellement **51 entreprises cotées à Bursa Malaysia**.
La source unique des noms, tickers et alias est
[`src/finagent/config/universe.py`](src/finagent/config/universe.py).
Market, Forecast et le scraping utilisent cet univers, y compris 99 Speed Mart.

## Produit final : FastAPI + mémoire RAM + Streamlit

Le prototype utilise les données actuelles, l'index RAG sauvegardé et le **LSTM
final sélectionné manuellement**. Les phases de reconstruction et d'entraînement
sont terminées ; le lancement du produit ne réexécute aucun notebook.

Terminal 1, depuis la racine :

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn finagent.api.main:app --reload --app-dir src
```

Terminal 2 :

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

Dashboard : <http://localhost:8501> · API : <http://127.0.0.1:8000/docs>.
Pages : **Market Overview**, **Company Intelligence**, **FinAgent AI**,
**Model & System Insights**. Gemini utilise la clé existante dans `src/.env`.

Voir le [guide produit](docs/product.md) pour les endpoints, les paramètres,
les limites de mémoire, la provenance des métriques et les smoke tests.
Les documents de Phase A/B restent des archives de développement.

## Règles de données

- Identification : noms et alias normalisés, tickers exacts, priorité aux noms
  spécifiques dans les chevauchements. Les agents individuels conservent leur
  périmètre société ; le chat orchestre des comparaisons entre deux sociétés.
- Scraping : sujet de paragraphe vérifié avec un matcher conservateur. Une entité
  incertaine ou extérieure à l'univers n'est pas attribuée à sa société mère.
- RAG : déduplication de la représentation de retrieval par
  `company + ticker + text_hash`, avec conservation de toutes les provenances.
  Les lignes sources destinées au sentiment et au Forecast restent distinctes.
- Récence : `recent`, `latest`, `recently` et `last weeks` désignent les 90 derniers
  jours relativement à la dernière publication du corpus. Une période explicite
  interprétable est prioritaire. Les mois et trimestres doivent préciser l'année.
  La fenêtre porte sur la date de publication, pas sur la période comptable citée.
  Une absence de document dans la fenêtre est signalée sans recours à des news anciennes.
- Volume : un volume précédent nul donne `volume_change = NaN`.
- Forecast : signal conceptuel à **18:00 Asia/Kuala_Lumpur**. Les news disponibles
  au plus tard à 18:00 sont utilisables pour ce signal ; les suivantes attendent le
  prochain signal observé du ticker. Les timestamps sources et le mapping sont conservés.
- La cible reste `target_return_1d = 100 × (Close[t+1] / Close[t] - 1)` par ticker,
  avec une fenêtre de 60 observations et le modèle global existant. 99 Speed Mart
  reste un cas potentiel de généralisation à une entreprise absente des séquences
  d'entraînement historiques ; aucun changement de split n'a été effectué pour elle.

## Tests locaux de Phase A

Depuis la racine du projet, avec l'environnement existant :

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s tests -v
```

Les tests utilisent des données synthétiques et des copies en mémoire des CSV
actuels. Ils n'appellent aucun LLM, ne téléchargent aucun modèle, ne construisent
aucun index FAISS et ne réécrivent aucun dataset. Les ressources des trois agents
sont chargées seulement lors d'un appel effectif nécessitant ces ressources.

Les anciennes consignes de poursuite de Phase B sont abandonnées. Les données et
artefacts présents constituent la base finale du prototype.
