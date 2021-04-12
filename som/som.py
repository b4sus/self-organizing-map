import time

import numpy as np


class SOM:

    def __init__(self, map_length, init_learning_rate=1, learning_rate_constant=3000, init_sigma=1.2,
                 sigma_constant=4000, max_iter=None, observer=None):
        """

        :param map_length: how many neurons in each column/row
        :param init_learning_rate: specifies how much will the neighboring neurons affected
        :param learning_rate_constant: specifies how fast will learning rate decrease
        :param init_sigma: specifies how big will the initial neighborhood will be - 1/4 of map_length seems ok
        :param sigma_constant: specifies how fast will learning rate decrease
        :param max_iter:maximum number of iterations
        :param observer: callable invoked after every iteration
        """
        self.map_length = map_length
        self.rng = np.random.default_rng()
        self.init_learning_rate = init_learning_rate
        self.learning_rate_constant = learning_rate_constant
        self.init_sigma = init_sigma
        self.sigma_constant = sigma_constant
        self.max_iter = max_iter if max_iter is not None else 500 * map_length * map_length
        self.observer = observer

    def fit(self, X, *args, **kwargs):
        nr_features = X.shape[1]
        self.Thetas = self.rng.uniform(size=(self.map_length, self.map_length, nr_features))
        self.iteration = 0
        learning_rate = self.init_learning_rate
        sigma = self.init_sigma

        self.coords = np.empty((self.map_length, self.map_length, 2))
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                self.coords[neuron_x, neuron_y] = np.array([neuron_x, neuron_y])

        while self.iteration < self.max_iter:
            x = X[self.rng.integers(0, len(X))]
            bmu_coords = self.find_bmu_using_np(x)
            Deltas, neighborhoods = self.find_deltas_using_np(x, bmu_coords, learning_rate, sigma, nr_features)

            self.Thetas += Deltas

            self.iteration += 1
            learning_rate = self.learning_rate()
            sigma = self.sigma()

            if self.observer is not None:
                self.observer(X=X, map_length=self.map_length, iter=self.iteration, Thetas=self.Thetas,
                              learning_rate=learning_rate, sigma=sigma,
                              neighbor_change=self.direct_neighbor(neighborhoods, bmu_coords),
                              X_repr=kwargs.get("X_repr", None))
        return self

    def transform(self, X, *args, **kwargs):
        return X

    def find_deltas_using_np(self, x, bmu_coords, learning_rate, sigma, nr_features):
        """
        For each neuron in map calculates its change given sample x, bmu coordinates and current learning_rate and sigma.
        Formula: change_for_theta_x_y = learning_rate * n_x_y * (x - theta_x_y)
        where n_x_y is result of neighborhood formula: exp(norm(bmu_coords - [x, y]]) ** 2 / (sigma ** 2))
        :param x:
        :param bmu_coords:
        :param learning_rate:
        :param sigma:
        :param nr_features:
        :return: matrix of changes for thetas, matrix of neighborhoods
        """
        D = np.linalg.norm(np.array(bmu_coords) - self.coords, axis=2)
        N = np.exp(-(D ** 2) / (sigma ** 2))
        Deltas = np.empty((self.map_length, self.map_length, nr_features))
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                Deltas[neuron_x, neuron_y] = learning_rate * N[neuron_x, neuron_y] * (x - self.Thetas[neuron_x, neuron_y])
        return Deltas, N

    def find_deltas_using_loops(self, x, bmu_coords, learning_rate, sigma, nr_features):
        """
        Easier to read, not vectorized (slower), version of find_deltas_using_np
        """
        Deltas = np.empty((self.map_length, self.map_length, nr_features))
        Neighborhoods = np.empty((self.map_length, self.map_length))
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                n = self.neighborhood(bmu_coords, (neuron_x, neuron_y), sigma)
                Neighborhoods[neuron_x, neuron_y] = n
                Deltas[neuron_x, neuron_y] = learning_rate * n * (x - self.Thetas[neuron_x, neuron_y])
        return Deltas, Neighborhoods

    def find_bmu_using_np(self, x):
        """
        Finds the best matching unit in map for given sample x
        """
        return np.unravel_index(np.argmin(np.linalg.norm(self.Thetas - x, axis=2)), (self.map_length, self.map_length))

    def find_bmu_using_loops(self, x):
        """
        Easier to read, not vectorized (slower), veresion of find_bmu_using_np
        """
        bmu_score = np.inf
        bmu_coords = None
        for neuron_x in range(self.map_length):
            for neuron_y in range(self.map_length):
                theta = self.Thetas[neuron_x, neuron_y]
                # score = np.linalg.norm(x - theta)
                # no need to root square - slightly faster
                score = ((x - theta) ** 2).sum()
                if score < bmu_score:
                    bmu_coords = (neuron_x, neuron_y)
                    bmu_score = score
        return bmu_coords

    def direct_neighbor(self, neighborhoods, bmu_coords):
        """
        Returns neighborhood value for direct neighbor - for debug purpose only
        """
        bmu_x, bmu_y = bmu_coords
        assert neighborhoods[bmu_x, bmu_y] == 1
        if bmu_x < self.map_length - 1:
            neighbor_x = bmu_x + 1
        elif bmu_x > 0:
            neighbor_x = bmu_x - 1
        return neighborhoods[neighbor_x, bmu_y]

    def learning_rate(self):
        return max(self.init_learning_rate * np.exp(-self.iteration / self.learning_rate_constant), 0.1)

    def neighborhood(self, bmu_coords, current_neuron, sigma):
        d = np.linalg.norm(np.array(bmu_coords) - np.array(current_neuron))
        return np.exp(-(d ** 2) / (sigma ** 2))

    def sigma(self):
        return max(self.init_sigma * np.exp(-self.iteration / self.sigma_constant), 1)


class TrainedSOM:
    def __init__(self, Thetas):
        self.Thetas = Thetas
        self.map_length = Thetas.shape[0]
        self.nr_of_features = Thetas.shape[2]

    def fit(self, *args):
        return self

    def transform(self, X):
        X_transformed = np.empty(shape=len(X), dtype=object)
        for i, x in enumerate(X):
            X_transformed[i] = np.unravel_index(np.argmin(np.linalg.norm(self.Thetas - x, axis=2)), (self.map_length,
                                                                                                     self.map_length))
        return X_transformed

    def predict(self, X):
        pass
