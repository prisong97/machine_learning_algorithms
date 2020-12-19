import numpy as np
from scipy import linalg


class gp_classification_model:
    def __init__(
        self, theta_0=1.0, theta_1=4.0, theta_2=1.0, theta_3=2.0, beta_inv=0.01
    ):
        np.random.seed(50)
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.beta_inv = beta_inv

    def sigmoid(self, z):
        """
        returns a value between 0 and 1, as given by the sigmoid
        activation function
        """

        return 1 / (1 + np.exp(-z))

    def construct_gram_element(self, x_i, x_j, theta_0, theta_1, theta_2, theta_3):
        """
        kernel element given by the exponential of a quadratic form (eqn 6.63, pg 307 of Bishop's PRML)
        """

        k_ij = (
            theta_0 * np.exp(-(theta_1 / 2) * np.power(np.linalg.norm(x_i - x_j), 2))
            + theta_2
            + theta_3 * np.dot(np.transpose(x_i), x_j)
        )
        return k_ij

    def construct_kernel(self, phi_matrix, beta_inv):
        """
        construct the NxN C_matrix, where N is the number of examples.
        Include beta_inv to ensure matrix is positive definite
        """

        no_of_examples, _ = phi_matrix.shape
        C_matrix = np.array([])

        for i in range(no_of_examples):
            x_i = phi_matrix[i]
            c_matrix_row = np.array([])

            for j in range(no_of_examples):
                x_j = phi_matrix[j]
                gram_element = self.construct_gram_element(
                    x_i, x_j, self.theta_0, self.theta_1, self.theta_2, self.theta_3
                )

                c_matrix_row = np.append(c_matrix_row, gram_element)
            C_matrix = np.concatenate((C_matrix, c_matrix_row))

        C_matrix = np.reshape(C_matrix, newshape=(no_of_examples, no_of_examples))

        return C_matrix + beta_inv * np.identity(no_of_examples)

    def construct_W_matrix(self, a_bar):
        """
        construct the diagonal W_matrix, where each entry is given by
        W_ii = sigma_i * (1-sigma_i)
        """

        no_of_examples = len(a_bar)
        W_matrix = np.identity(no_of_examples)

        for i in range(no_of_examples):
            a = a_bar[i]
            W_matrix[i, i] = self.sigmoid(a) * (1 - self.sigmoid(a))

        return W_matrix

    def compute_hessian(self, C_matrix, W_matrix):
        """
        computes the second derivative of E(w), returning an MxM matrix where
        M is the projected dimension of the data
        """

        Hessian = linalg.inv(C_matrix) + W_matrix
        return Hessian

    def compute_dE(self, t_bar, C_matrix, a_init):
        """
        computes the first derivative of E(w), returning an Mx1 vector where
        M is the projected dimension of the data
        """

        # compute sigma_bar
        sigma_bar = np.array([])
        for i in range(len(t_bar)):
            sigma_bar = np.append(sigma_bar, self.sigmoid(a_init[i]))

        dE = t_bar - sigma_bar - np.matmul(linalg.inv(C_matrix), a_init)
        return dE

    def newton_raphson_update(self, a_init, phi_matrix, t_bar, iterations):
        """
        main function to be run
        """

        # construct C_matrix
        self.C_matrix = self.construct_kernel(phi_matrix, self.beta_inv)

        for i in range(iterations):

            # construct W_matrix
            self.W_matrix = self.construct_W_matrix(a_init)

            Hessian = self.compute_hessian(self.C_matrix, self.W_matrix)
            dE = self.compute_dE(t_bar, self.C_matrix, a_init)

            # perform update
            a_new = a_init + np.matmul(linalg.inv(Hessian), dE)
            a_init = a_new

        # check for convergence
        convergence_diag = linalg.norm(a_new - a_init)
        if convergence_diag < 0.0001:
            print(f"The estimate has converged after {iterations} iterations.")
        else:
            print(f"The estimate has not converged after {iterations} iterations.")

        return a_init

    def gp_classification_pred(self, mu_star, s_star):

        numer = mu_star
        denom = (1 + (np.pi / 8) * (s_star ** 2)) ** (1 / 2)
        return self.sigmoid(numer / denom)

    def gp_prediction_by_batch(self, a_bar, C_matrix, W_matrix, phi_matrix):

        no_of_examples, _ = phi_matrix.shape
        gp_pred_batch = np.array([])

        for i in range(no_of_examples):

            x_i = phi_matrix[i]

            gp_pred = self.gp_prediction_individual(
                a_bar, C_matrix, W_matrix, phi_matrix, x_i
            )

            gp_pred_batch = np.append(gp_pred_batch, gp_pred)

        return gp_pred_batch

    def gp_prediction_individual(self, a_bar, C_matrix, W_matrix, phi_matrix, x_mat):

        no_of_examples, _ = phi_matrix.shape

        # construct k_star_vector
        k_star_vector = np.array([])

        for j in range(no_of_examples):
            x_j = phi_matrix[j]
            k_entry = self.construct_gram_element(
                x_mat, x_j, self.theta_0, self.theta_1, self.theta_2, self.theta_3
            )
            k_star_vector = np.append(k_star_vector, k_entry)

        k_star_vector = np.reshape(k_star_vector, newshape=(no_of_examples, 1))

        # construct c_star
        c_star = (
            self.construct_gram_element(
                x_mat, x_mat, self.theta_0, self.theta_1, self.theta_2, self.theta_3
            )
            + self.beta_inv
        )

        # compute mu_star
        mu_star_aux = np.matmul(np.transpose(k_star_vector), linalg.inv(C_matrix))
        mu_star = np.matmul(mu_star_aux, a_bar)

        # compute s_star
        s_star_aux = np.matmul(
            np.transpose(k_star_vector), linalg.inv((linalg.inv(W_matrix) + C_matrix))
        )
        s_star = c_star - np.matmul(s_star_aux, k_star_vector)

        gp_pred = self.gp_classification_pred(mu_star, s_star)

        return gp_pred
