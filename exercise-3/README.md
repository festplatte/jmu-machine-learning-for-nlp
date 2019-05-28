# Train IMDB

Trainiert ein neuronales Netzwerk um Filmbewertungen nach positiv oder negativ zu sortieren. Zum Optimieren von Hyperparametern wurde der Ordner `imdb-data/train` in 90 % Trainingsdaten (`imdb-data/train`) und 10 % Validationdaten (`imdb-data/validation`) aufgeteilt. Es wurden dabei jeweils die ersten 10 % der Daten als Validationdaten verwendet.

## Verwendung

- Download der Daten von IMDB über http://ai.stanford.edu/~amaas/data/sentiment/
- Extrahieren in Unterverzeichnis `imdb-data`
- Aufteilen von `imdb-data/train` in Trainings- und Validationdaten
- Ausführen von `python3 main.py`
