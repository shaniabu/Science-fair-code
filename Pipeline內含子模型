
import pandas as pd
df = pd.read_csv('kidney_data-2.csv', header = None)

from sklearn.preprocessing import LabelEncoder

#0 is ID
#1-3 is feature
#4 is result

x = df.loc[:, 1:3].values
y = df.loc[:, 4].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

le.transform(['A', 'B', 'C', 'E', 'F'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(x, y, 
                     test_size=0.2,
                     stratify=y,
                     random_state=1)

#use Pipeline to train the model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1))

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
#Test Accuracy: 0.870

#use k-folder 

import numpy as np
from sklearn.model_selection import StratifiedKFold 


kfold = StratifiedKFold(n_splits=10,
                        random_state=1).split(X_train, y_train)

scores = []
for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %2d, Class dist.: %s, Acc: %.3f' % (k+1,
          np.bincount(y_train[train]), score))


k_range = range(1,11)
import matplotlib.pyplot as plt
plt.plot(k_range, scores)
plt.xlabel('iteration number')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

#CV accuracy: 0.869 +/- 0.071
#15-CV accuracy: 0.864 +/- 0.038
#10-CV accuracy: 0.863 +/- 0.033



#draw the learning curve
from sklearn.model_selection import learning_curve
pipe_lr = make_pipeline(StandardScaler(),
                        LogisticRegression(penalty='l2', random_state=1))

train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
                               X=X_train,
                               y=y_train,
                               train_sizes=np.linspace(0.1, 1.0, 20),
                               cv=20,
                               n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='red', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='red')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 1])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi=300)
plt.show()

#驗證曲線是非常有用的工具，他可以用來提高模型的性能，原因是他能處理過擬合和欠擬合問題。
#驗證曲線和學習曲線很相近，不同的是這裏畫出的是不同參數下模型的準確率而不是不同訓練集大小下的準確率：

#畫驗證曲線的圖


from sklearn.model_selection import validation_curve
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(
                estimator=pipe_lr, 
                X=X_train, 
                y=y_train, 
                param_name='logisticregression__C', 
                param_range=param_range,
                cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, 
         color='blue', marker='o', 
         markersize=5, label='training accuracy')

plt.fill_between(param_range, train_mean + train_std,
                 train_mean - train_std, alpha=0.15,
                 color='blue')

plt.plot(param_range, test_mean, 
         color='red', linestyle='--', 
         marker='s', markersize=5, 
         label='validation accuracy')

plt.fill_between(param_range, 
                 test_mean + test_std,
                 test_mean - test_std, 
                 alpha=0.15, color='red')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.7, 1])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi=300)
plt.show()



# ## Reading a confusion matrix


from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)




fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
#plt.savefig('images/06_09.png', dpi=300)
plt.show()

#計算宏觀平均的F1

from sklearn.metrics import precision_recall_fscore_support
p_class, r_class, f_class, support_micro=precision_recall_fscore_support(
                y_true=y_test, y_pred=y_pred, labels=[0, 1, 2, 3, 4], average=None)
print('各類單獨F1:',f_class)
print('宏F1：',f_class.mean())
#各類單獨F1: [0.84444444 0.         0.85714286 1.         0.         0.        ]
#宏F1： 0.4502645502645503
from sklearn.metrics import f1_score
print(f1_score(y_true=y_test, y_pred=y_pred, average='macro'))
print(f1_score(y_true=y_test, y_pred=y_pred, average='micro'))



#0.5403174603174603
#0.8985507246376812

