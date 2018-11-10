import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('091118Data_S2-DATA_JABON_Foto_Jabon_Run.csv')
X = dataset.iloc[:, :2]
y = dataset.Class

for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = svm.SVC(kernel='rbf', C=1, gamma=0.125)
    classifier.fit(X_train, y_train)
    print(classifier.score(X_test, y_test))

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('DeepSkyBlue', 'yellow', 'Chartreuse', 'pink')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('blue', 'orange', 'green', 'red'))(i), label=j)
    plt.title('SVM (Training set)' + str(classifier.score(X_test, y_test)))
    plt.xlabel('Energy')
    plt.ylabel('Entropy')
    plt.legend()
    plt.show()
    # ----- Coufusion Matriks
    y_pred = classifier.predict(X_test)
    a = confusion_matrix(y_test, y_pred)
    print(a)
    # print(X_train)