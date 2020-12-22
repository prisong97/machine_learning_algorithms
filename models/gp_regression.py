import matplotlib.pyplot as plt
import numpy as np


def construct_gram_element(x_i, x_j, theta_0, theta_1, theta_2, theta_3):
    """
    kernel element given by the exponential of a quadratic form (eqn 6.63, pg 307 of Bishop's PRML).
    """
    k_ij = (
        theta_0 * np.exp(-(theta_1 / 2) * np.power(np.linalg.norm(x_i - x_j), 2))
        + theta_2
        + theta_3 * np.dot(np.transpose(x_i), x_j)
    )
    return k_ij


def gaussian_process_regression(x, y, t, beta_inv, theta_0, theta_1, theta_2, theta_3):
    """ 
    Computes the Gaussian process prediction of every input data sequentially, 
    where each input data is taken randomly from the noise data.
    """
    np.random.seed(355)
    random_indices = np.random.permutation(np.array(range(len(x))))
    x_ordered = x[random_indices]
    counter = 0

    t_seen = np.array([])
    x_seen = np.array([])

    # initialise the c_matrix
    first_index = random_indices[0]
    k_element = construct_gram_element(
        x[first_index], x[first_index], theta_0, theta_1, theta_2, theta_3
    )
    c_matrix = np.array([np.array([k_element]) + beta_inv])

    for i in random_indices:

        # compute c_matrix inv
        c_matrix_inv = np.linalg.inv(c_matrix)

        x_new = x[i]
        t_new = t[i]

        t_seen = np.append(t_seen, t_new)
        x_seen = np.append(x_seen, x_new)
        seen_data_length = len(x_seen)

        mu_list = np.array([])
        cov_list = np.array([])

        for j in random_indices:

            x_0 = x[j]
            k_star = np.array([])
            c_star = (
                np.array(
                    [
                        construct_gram_element(
                            x_0, x_0, theta_0, theta_1, theta_2, theta_3
                        )
                    ]
                )
                + beta_inv
            )

            for n in range(seen_data_length):
                data_n = x_seen[n]
                k_element = construct_gram_element(
                    x_0, data_n, theta_0, theta_1, theta_2, theta_3
                )
                k_star = np.append(k_star, k_element)

            # construct c_star_new arguments
            if counter < len(x) - 1:
                next_data_index = random_indices[counter + 1]
                if j == next_data_index:
                    k_star_matrix = np.transpose(np.array([k_star]))
                    c_star_matrix = np.array([c_star])

                    c_temp_row_1 = np.concatenate((c_matrix, k_star_matrix), axis=1)
                    c_temp_row_2 = np.concatenate(
                        (np.transpose(k_star_matrix), c_star_matrix), axis=1
                    )
            else:
                # last iteration -- c_temp_row_2 does not matter because
                # we won't be using the c_matrix beyond this iteration.
                c_temp_row_1 = c_matrix

            # construct mu_star
            mu_star = np.matmul(np.matmul(np.transpose(k_star), c_matrix_inv), t_seen)

            # construct cov_star
            cov_star = c_star - np.matmul(
                np.matmul(np.transpose(k_star), c_matrix_inv), k_star
            )

            mu_list = np.append(mu_list, mu_star)

            cov_list = np.append(cov_list, np.sqrt(cov_star))

        # new c_matrix
        c_matrix = np.concatenate((c_temp_row_1, c_temp_row_2))

        plt.figure(figsize=(10, 5))
        plt.title(f"Plot after seeing {(counter+1)} points.")
        plt.scatter(x, t, c="green")
        plt.plot(x, y, c="red", label="original curve")
        plt.errorbar(x_ordered, mu_list, yerr=cov_list, linestyle="none", marker="s")
        plt.legend()

        plt.show()

        counter += 1

        print(random_indices[:counter])
