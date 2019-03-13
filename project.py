#sciagamy wszystkie potrzebne paczuszki

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


#importujemy dane z pliku csv
data = pd.read_csv('creditcard.csv')

#przegladamy co mamy tak naprawde w zestawie danych
print(data.columns)

# Print the shape of the data
data = data.sample(frac=0.5, random_state = 1)
print(data.shape)
print(data.describe())

# V1 - V28
# SĄ TO WYNIKI  tzw PCA Dimensionality reduction, ktore służą do ochrony danych osobowych uzytkownika



#Sprawdzamy liczbe oszustw spośród danych

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]


#Jaka czesc wszystkich transakcji okazuje sie oszustwem i wypisyjemy
outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print('Przypadki oszustwa: {}'.format(len(data[data['Class'] == 1])))
print('Dobre transakcje: {}'.format(len(data[data['Class'] == 0])))



# Tworzymy macierz korelacji, zeby spradzic czy istnieja jakies zaleznosci
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()



#Wyciagamy wszystkie kolumny.
columns = data.columns.tolist()

#Filtrujemy po kolumnach, aby usunac dane ktore nas nie interesuja.
columns = [c for c in columns if c not in ["Class"]]

#wartosc ktora bedziemy przewidywac
target = "Class"

X = data[columns]
Y = data[target]

# Print shapes
print(X.shape)
print(Y.shape)



#Konieczny argument, nie wiem dokladnie po co
state = 1

# Dwie metody przewidywania z ktorych bedziemy korzystac:
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction)}

# Wpasowanie do modelu
plt.figure(figsize=(9, 7))
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)

        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)

    # Przeksztalcamy zeby dobrze wypisywalo
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != Y).sum()

    print('{}: {}'.format(clf_name, n_errors))
    print("Accuracy score:",accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))
