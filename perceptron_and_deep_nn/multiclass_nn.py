import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

def ploter(X, ax, model):
    x = np.linspace(min(X[:, 0])-1, max(X[:, 0])+1, 50)
    y = np.linspace(min(X[:, 1])-1, max(X[:, 1])+1, 50)
    xx, yy = np.meshgrid(x, y)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    prediction = model.predict_classes(grid) # multiclass prediction
    z = prediction.reshape(xx.shape)
    ax.contourf(xx, yy, z)
    return ax

def prediction(x, y, model):
    return model.predict_classes(np.array([[x, y]]))

def main():
    n_pts = 500
    np.random.seed(0)
    centers = [[-1, 1], [-1, -1], [1, -1], [1, 1], [0, 0]]
    X, Y = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)
    Y_cat = to_categorical(Y, 5)

    model = Sequential()
    model.add(Dense(units=5, input_shape=(2,), activation='softmax')) # softmax for multiclass classification
    model.compile(Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(x=X, y=Y_cat, verbose=1, batch_size=50, epochs=100, shuffle='true')

    fig, ax = plt.subplots()
    ax = ploter(X,ax, model)
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1])
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1])
    ax.scatter(X[Y == 2, 0], X[Y == 2, 1])
    ax.scatter(X[Y == 3, 0], X[Y == 3, 1])
    ax.scatter(X[Y == 4, 0], X[Y == 4, 1])


    x, y = 0.5, 0.5
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
