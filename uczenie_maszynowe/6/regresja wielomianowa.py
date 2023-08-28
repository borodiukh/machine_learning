import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing


alldata = pandas.read_csv('data6.tsv', sep=',', names=())

X = np.array(alldata['X']).reshape(-1, 1)
Y = np.array(alldata['Y'])
# print(X)
# print(Y)


# wielomian pierwszego stopnia

poly = preprocessing.PolynomialFeatures(1, include_bias=False)
poly_features = poly.fit_transform(X)

model_1 = linear_model.LinearRegression()
model_1.fit(poly_features, Y)

x_poly = np.arange(10, 200).reshape(-1, 1)
x_poly = poly.transform(x_poly)

pred_y = model_1.predict(x_poly)
plt.title('Linear regression with polynomial feauteres 1')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,Y)
plt.plot(x_poly[:, 0], pred_y)
plt.show()


# wielomian drugiego stopnia

poly = preprocessing.PolynomialFeatures(2, include_bias=False)
poly_features = poly.fit_transform(X)
model_1 = linear_model.LinearRegression()
model_1.fit(poly_features, Y)
x_poly = np.arange(10, 200).reshape(-1, 1)
x_poly = poly.transform(x_poly)

pred_y = model_1.predict(x_poly)
plt.title('Linear regression with polynomial feauteres 2')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,Y)
plt.plot(x_poly[:, 0], pred_y)
plt.show()


# wielomian piatego stopnia

poly = preprocessing.PolynomialFeatures(5, include_bias=False)
poly_features = poly.fit_transform(X)
model_1 = linear_model.LinearRegression()
model_1.fit(poly_features, Y)
x_poly = np.arange(10, 200).reshape(-1, 1)
x_poly = poly.transform(x_poly)

pred_y = model_1.predict(x_poly)
plt.title('Linear regression with polynomial feauteres 5')
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,Y)
plt.plot(x_poly[:, 0], pred_y)
plt.show()