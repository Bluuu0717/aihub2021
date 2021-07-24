# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # import matplotlib as mpl
# # import math
# import seaborn as sns
#
# color = sns.color_palette()[8]

# #%%
# auto["loss_flag"]= auto.loss.map(lambda x:1 if x>0 else )
# #%%
# auto.loss_flag.value_count()
# #%%
# fig = plt.figure()
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)
# auto.boxplot(column="age")

# import numpy as np
# import matplotlib.pyplot as plt
#
# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# C, S = np.cos(X), np.sin(X)
#
# plt.plot(X,C)
# plt.plot(X,S)
#
# plt.show()

# %matplotlib inline
#图在里面
# if __name__ == '__main__':
#     plt.figure()
#     plt.plot(np.linspace(0, np.pi * 2), np.sin(np.linspace(0, np.pi * 2)), color=color)
#     plt.show()


import operator
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
#
# x_ = np.load('mystery2-x.npy')
# y_ = np.load('mystery2-y.npy')
#
# print(x_.shape, y_.shape)
#
#
#
# x = x_.reshape(-1, 1)
# y = y_.reshape(-1, 1)
#
#
# plt.figure(figsize=(7,5))
# plt.scatter(x, y, edgecolor = 'k',label='data',color = '#B9D6D6')
# plt.xlabel('x',fontsize=16)
# plt.ylabel('y',fontsize=16)
# plt.legend(fontsize=16)
#
# plt.show()
#
# x_train, x_val, y_train, y_val = train_test_split(x,y,train_size=0.75,random_state=1)
#
# maxdeg = 20
# training_error, validation_error = [],[]
# #
# for d in range(maxdeg):
#
#     x_poly_train = PolynomialFeatures(d).fit_transform(x_train)
#     x_poly_val = PolynomialFeatures(d).fit_transform(x_val)
#
#     lreg = LinearRegression()
#     lreg.fit(x_poly_train, y_train)
#
#     y_train_pred = lreg.predict(x_poly_train)
#     y_val_pred = lreg.predict(x_poly_val)
#
#
#     training_error.append(mean_squared_error(y_train, y_train_pred))
#     validation_error.append(mean_squared_error(y_val, y_val_pred))
#
#     min_mse = min(validation_error)

#
# best_degree = validation_error.index(min_mse)
#
# print("The best degree of the model is", best_degree)

#
# poly = PolynomialFeatures(degree = 20)
# X_poly = poly.fit_transform(x)
#
# poly.fit(X_poly, y)
# lin2 = LinearRegression()
# lin2.fit(X_poly, y)





import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


from sklearn.linear_model import LinearRegression
import numpy as np

x = np.linspace(0, 1)
y = 3 * x + 4 + np.random.rand() / 100
x = x.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)





















