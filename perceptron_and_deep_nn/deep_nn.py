import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def ploter(X, ax, model):
    x = np.linspace(min(X[:, 0])-1, max(X[:, 0])+1)
    y = np.linspace(min(X[:, 1])-1, max(X[:, 1])+1)
    xx, yy = np.meshgrid(x, y)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    prediction = model.predict(grid)
    z = prediction.reshape(xx.shape)
    ax.contourf(xx, yy, z)
    return ax

def prediction(x, y, model):
    return model.predict(np.array([[x, y]]))

def main():
    n_pts = 500
    np.random.seed(0)
    X, Y = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)

    model = Sequential()
    model.add(Dense(4, input_shape=(2,), activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
    h = model.fit(x=X, y=Y, verbose=1, batch_size=20, epochs=500, shuffle='true')

    fig, ax = plt.subplots()
    ax = ploter(X,ax, model)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1])
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1])
    x, y = 0.1, 0.25
    print(prediction(x, y, model))
    ax.plot([x], [y], marker='o', markersize=10, color='red')
    plt.show()

    plt.plot(h.history['accuracy'])
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy'])
    plt.show()

    plt.plot(h.history['loss'])
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'])
    plt.show()
if __name__ == "__main__":
    main()
