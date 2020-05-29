import pandas as pd

from sklearn.preprocessing import LabelEncoder
import numpy as np





df = pd.read_csv('super_input2.csv', header = None)

x1 = df.loc[:, 0:1].values
x2 = df.loc[:, 2:3].values
y = df.loc[:, 4:12].values


a=y[:, 0]
a_list=a.tolist()
i=0
while i<len(a_list):
    if a_list[i]>1500:
        a_list[i]=int(100-(100-a_list[i]/6000)*2/3)
    else:
        a_list[i]=int((a_list[i])/3000)
    i=i+1
    
b=y[:,7]
b_list=b.tolist()
i=0
while i<len(b_list):
    if b_list[i]>5:
        b_list[i]=int((100-b_list[i])*0.5/0.95)
    else:
        b_list[i]=int((100-b_list[i]*10))
    i=i+1
    

a_array=np.array(a_list)
b_array=np.array(b_list)

c_array=y[:,2:6]
d_array=y[:,8]
e_array=np.column_stack((c_array, d_array))

e_array=np.column_stack((a_array, e_array))
input_data=np.column_stack((e_array, b_array))
y=input_data

c=x2[:, 0]
le = LabelEncoder()
c = le.fit_transform(c)
d=x2[:, 1]
d = le.fit_transform(d)

datadata=np.column_stack((c, d))
x=np.column_stack((x1, datadata))


y=np.mean(y, axis=1)

y_list=y.tolist()
j=0
while j<=807:
    y_list[j]=int(y_list[j])
    j=j+1

y=np.array(y_list)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(x, y, 
                     test_size=0.2,
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
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test)