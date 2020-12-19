import numpy as np
from scipy import linalg


class newton_raphson_update:
    """
    this class obtains the MAP estimate of w_bar via the newton raphson algorithm.
    """

    def __init__(self):
        np.random.seed(50)

    def sigmoid(self, z):
        """
        returns a value between 0 and 1, as given by the sigmoid
        activation function
        """
        return 1 / (1 + np.exp(-z))

    def compute_dE(self, phi_matrix, y_vector, t_vector):
        """
        computes the first derivative of E(w), returning an Mx1 vector where
        M is the projected dimension of the data
        """

        dE = np.matmul(np.transpose(phi_matrix), (y_vector - t_vector))
        return dE

    def get_R_matrix(self, y_vector):
        """
        compute diagonal NxN matrix R, where
        R_nn = y_n*(1-y_n)
        """
        R = np.identity(len(y_vector))

        for i in range(len(y_vector)):
            y_n = y_vector[i]
            R[i, i] = y_n * (1 - y_n)

        return R

    def compute_hessian(self, phi_matrix, y_vector):
        """
        computes the second derivative of E(w), returning an MxM matrix where
        M is the projected dimension of the data
        """
        R = self.get_R_matrix(y_vector)

        Hessian = np.matmul(np.matmul(np.transpose(phi_matrix), R), phi_matrix)
        return Hessian

    def update_weights(self, alpha_inv, w_init, phi_matrix, t_vector, iterations):
        """
        main function to run for performing newton-raphson updating
        of weights
        """

        for i in range(iterations):

            # get dE
            y_vector = np.array([])
            a_vector = np.matmul(phi_matrix, w_init)

            for a in a_vector:
                y_val = self.sigmoid(a)
                y_vector = np.append(y_vector, y_val)

            y_vector = np.reshape(y_vector, (len(y_vector), 1))

            dE_ = np.matmul(
                alpha_inv * np.identity(len(w_init)), w_init
            ) + self.compute_dE(phi_matrix, y_vector, t_vector)

            # get Hessian
            hessian_ = alpha_inv * np.identity(len(w_init)) + self.compute_hessian(
                phi_matrix, y_vector
            )

            # perform update
            w_new = w_init - np.matmul(linalg.inv(hessian_), dE_)

            w_init = w_new

        # check for convergence
        convergence_diag = linalg.norm(w_new - w_init)
        if convergence_diag < 0.0001:
            print(f"The estimate has converged after {iterations} iterations.")
        else:
            print(f"The estimate has not converged after {iterations} iterations.")

        return w_init, y_vector


class predictive_distribution:
    """
    this class computes the probability of t_star given t_bar and x_star
    using w_bar computed above.
    """

    def __init__(self):
        np.random.seed(50)

    def sigmoid(self, z):
        """
        returns a value between 0 and 1, as given by the sigmoid
        activation function
        """
        return 1 / (1 + np.exp(-z))

    def compute_t_star(self, w_map, S_star, x_star):
        """
        computes p(t_star|t_bar, x_star) per observation
        """

        numer = np.matmul(np.transpose(w_map), x_star)
        denom = (1 + (np.pi / 8) * S_star) ** (1 / 2)
        pred = self.sigmoid(numer / denom)

        return pred

    def compute_predictions(self, w_map, S_n, phi_matrix):
        """
        computes p(t_star|t_bar, x_star) for all the observations
        in the phi_matrix
        """

        t_star_pred = np.array([])
        for x_star in phi_matrix:
            s_star = np.matmul(np.matmul(np.transpose(x_star), S_n), x_star)

            t_star = self.compute_t_star(w_map, s_star, x_star)
            t_star_pred = np.append(t_star_pred, t_star)

        return t_star_pred

    def compute_assigned_class(self, t_star_predictions):
        """
        computes the assigned class of each prediction, where p >=0.5
        corresponds to class 1, and p<0.5 to class 0.
        """

        class_pred = np.array([])
        for t_pred in t_star_predictions:
            if t_pred >= 0.5:
                assigned_class = 1
            else:
                assigned_class = 0
            class_pred = np.append(class_pred, assigned_class)

        return class_pred
