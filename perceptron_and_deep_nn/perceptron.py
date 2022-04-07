import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
    ln = plt.plot(x1,x2, '-')
    plt.pause(0.0001)
    #ln[0].remove()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def calculate_error(line_parameters, points, y):
    sigma = points * line_parameters
    p = sigmoid(sigma)
    one = np.ones_like(y)
    n = np.shape(one)[0]
    cross_entropy = -(np.log(p).T * y + np.log(one-p).T*(one-y))/n
    return cross_entropy

def gd(line_parameters, points, y, alpha):
    count = 0
    e = 10
    m = points.shape[0]
    while count < 4000:# or e < 0.01:
        count += 1
        p = sigmoid(points*line_parameters)
        g = (points.T * (p - y))/ m
        line_parameters = line_parameters - alpha * g
        w1, w2, b= line_parameters.item(0), line_parameters.item(1), line_parameters.item(2)
        x1 = np.array([points[:,0].min(), points[:,0].max()])
        x2 = -b / w2 + x1 * (-w1 /w2)
        e = calculate_error(line_parameters, points, y)
    draw(x1, x2)

def main():
    n_pts = 100
    np.random.seed(0)
    bias = np.ones(n_pts)
    top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).transpose()
    bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).transpose()
    all_points = np.vstack((top_region, bottom_region))
    line_parameters = np.matrix(np.zeros(3)).T
    y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape((-1, 1))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
    ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')

    gd(line_parameters,all_points, y, 0.06)
    plt.show()

if __name__ == "__main__":
    main()