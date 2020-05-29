import pandas as pd

import numpy as np



df = pd.read_csv('super_input3.csv', header = None)

x = df.loc[:, 0].values
y = df.loc[:, 1:9].values


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


y=np.mean(y, axis=1)

y_list=y.tolist()
j=0
while j<=807:
    y_list[j]=int(y_list[j])
    j=j+1
    
y=np.array(y_list)


x=x.reshape(-1, 1)
y=y.reshape(-1, 1)


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt



regr = LinearRegression()

# create quadratic features
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(x)
X_cubic = cubic.fit_transform(x)

# fit features
X_fit = np.arange(x.min(), x.max(), 1)[:, np.newaxis]

regr = regr.fit(x, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(x))

regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

plt.scatter(x, y, label='training points', color='lightgray')

plt.plot(X_fit, y_lin_fit, 
         label='linear (d=1), $R^2=%.2f$' % linear_r2, 
         color='blue', 
         lw=2, 
         linestyle=':')

plt.plot(X_fit, y_quad_fit, 
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red', 
         lw=2,
         linestyle='-')

plt.plot(X_fit, y_cubic_fit, 
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green', 
         lw=2, 
         linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='upper right')

#plt.savefig('images/10_11.png', dpi=300)
plt.show()







