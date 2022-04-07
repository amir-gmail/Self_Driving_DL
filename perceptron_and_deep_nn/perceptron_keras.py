import numpy as np
import matplotlib.pyplot as plt
import keras
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
    n_pts = 100
    np.random.seed(0)
    top_region = np.array([np.random.normal(13, 2, n_pts), np.random.normal(12, 2, n_pts)]).transpose()
    bottom_region = np.array([np.random.normal(8, 2, n_pts), np.random.normal(6, 2, n_pts)]).transpose()
    X = np.vstack((top_region, bottom_region))
    Y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape((-1, 1))

    model = Sequential()
    model.add(Dense(units=1, input_shape=[2], activation='sigmoid'))
    adam = Adam(lr=0.1)
    model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(x=X, y=Y, verbose=1, batch_size=50, epochs=500, shuffle='true')

    fig, ax = plt.subplots(figsize=(4, 4))
    ax = ploter(X, ax, model)
    ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
    ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
    x, y = 3, 4
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