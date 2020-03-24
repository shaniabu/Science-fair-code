import pandas as pd
df = pd.read_csv('kidney_data-1.csv', header = None)

from sklearn.preprocessing import LabelEncoder


x = df.loc[:, 1:2].values
y = df.loc[:, 3].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, stratify=y)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)


import matplotlib.pyplot as plt
import numpy as np




def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        x_test, y_test = x[test_idx, :], y[test_idx]

        plt.scatter(x_test[:, 0],
                    x_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100, 
                    label='test set')

x_combined_std = np.vstack((x_train_std, x_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.1, random_state=0, n_iter=1000)
ppn.fit(x_train_std, y_train)



lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(x_train_std, y_train)
plot_decision_regions(x_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
plt.xlabel('mutation position [standardized]')
plt.ylabel('mutation type [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_06.png', dpi=300)
plt.show()




from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(x_train_std, y_train)

plot_decision_regions(x_combined_std, 
                      y_combined,
                      classifier=svm, 
                      test_idx=range(105, 150))
plt.xlabel('mutation position [standardized]')
plt.ylabel('mutation type [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_11.png', dpi=300)
plt.show()




svm = SVC(kernel='rbf', random_state=1, gamma=0.2, C=1.0)
svm.fit(x_train_std, y_train)

plot_decision_regions(x_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('mutation position [standardized]')
plt.ylabel('mutation type [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_15.png', dpi=300)
plt.show()




svm = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
svm.fit(x_train_std, y_train)

plot_decision_regions(x_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('mutation position [standardized]')
plt.ylabel('mutation type [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_16.png', dpi=300)
plt.show()



svm = SVC(kernel='rbf', random_state=1, gamma=50.0, C=1.0)
svm.fit(x_train_std, y_train)

plot_decision_regions(x_combined_std, y_combined, 
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('mutation position [standardized]')
plt.ylabel('mutation type [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_16.png', dpi=300)
plt.show()


