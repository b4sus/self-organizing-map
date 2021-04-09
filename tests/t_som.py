import numpy as np
import matplotlib.pyplot as plt
import time

from som.som import SOM


def draw2(X, map_length, iter, Thetas, learning_rate, sigma,  neighbor_change=-1, **kwargs):
    if iter % 10 != 0:
        return
    plt.figure(1).clear()
    plt.title(
        f"Iteration {iter} - learning rate {learning_rate:.3f}, sigma {sigma:.3f}, "
        + f"neighbor_change {neighbor_change:.3f}")
    plt.scatter(X[:, 0], X[:, 1], marker="x")
    # if selected_x is not None:
    #     plt.scatter(selected_x[0], selected_x[1], marker="x", c="r")
    plt.scatter(Thetas[:, :, 0], Thetas[:, :, 1])
    # if winner_neuron is not None:
    #     plt.scatter(winner_neuron[0], winner_neuron[1], c="g")
    for neuron_x in range(map_length):
        for neuron_y in range(map_length):
            x = Thetas[neuron_x, neuron_y, 0]
            y = Thetas[neuron_x, neuron_y, 1]
            # plt.annotate(f"{neuron_x} {neuron_y}", (x, y))
            if neuron_x > 0:
                plt.plot([x, Thetas[neuron_x - 1, neuron_y, 0]], [y, Thetas[neuron_x - 1, neuron_y, 1]],
                         "k")
            if neuron_x < map_length - 1:
                plt.plot([x, Thetas[neuron_x + 1, neuron_y, 0]], [y, Thetas[neuron_x + 1, neuron_y, 1]],
                         "k")
            if neuron_y > 0:
                plt.plot([x, Thetas[neuron_x, neuron_y - 1, 0]], [y, Thetas[neuron_x, neuron_y - 1, 1]],
                         "k")
            if neuron_y < map_length - 1:
                plt.plot([x, Thetas[neuron_x, neuron_y + 1, 0]], [y, Thetas[neuron_x, neuron_y + 1, 1]],
                         "k")
    plt.figure(1).canvas.flush_events()
    plt.show(block=False)
    time.sleep(0.01)


def square_5():
    km = SOM(5, observer=draw2)
    rng = np.random.default_rng()
    X = rng.uniform(size=(1000, 2))

    km.fit(X)


def square_10():
    km = SOM(10, init_learning_rate=1, learning_rate_constant=3000, init_sigma=2.5, sigma_constant=3000, observer=draw2)
    rng = np.random.default_rng()
    X = rng.uniform(size=(1000, 2))

    km.fit(X)


def square_20():
    km = SOM(20, init_learning_rate=1, learning_rate_constant=12000, init_sigma=5, sigma_constant=12000, observer=draw2)
    rng = np.random.default_rng()
    X = rng.uniform(size=(10000, 2))

    km.fit(X)


def square_regions():
    km = SOM(20, init_learning_rate=1, learning_rate_constant=10000, init_sigma=4, sigma_constant=5000, observer=draw2)
    rng = np.random.default_rng()
    x1 = rng.normal(loc=.35, scale=0.1, size=(800, 1))
    x2 = rng.normal(loc=.8, scale=0.05, size=(200, 1))
    x = np.vstack((x1, x2))

    y = np.vstack((rng.permutation(x1), rng.permutation(x2)))

    km.fit(np.hstack((x, y)))


if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    square_20()
