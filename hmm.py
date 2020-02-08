from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their iex in B
        - state_dict: (num_state*1) A dictionary mapping each state to their iex in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here

        alpha[:, 0] = self.pi * self.B[:, self.obs_dict[Osequence[0]]]
        A_t = self.A.transpose()
        for i in range(1, L):
            temp = np.dot(A_t, alpha[:, i - 1])
            alpha[:, i] = temp * self.B[:, self.obs_dict[Osequence[i]]]
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        beta[:, L - 1] = 1
        for i in range(L - 2, -1, -1):
            B_t = self.B[:, self.obs_dict[Osequence[i + 1]]] * beta[:, i + 1]
            beta[:, i] = np.dot(self.A, B_t)
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = np.sum(alpha[:, -1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)
        temp = self.sequence_prob(Osequence)

        prob = alpha * beta / temp

        ###################################################
        return prob

    # TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        beta = self.backward(Osequence)
        alpha = self.forward(Osequence)

        for i in range(0, L - 1):
            for j in range(S):
                for k in range(S):
                    prob[j, k, i] = ((alpha[j, i] * self.A[j, k] * self.B[k, self.obs_dict[Osequence[i + 1]]] * beta[k, i + 1]) / self.sequence_prob(Osequence))
                    ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        # S = len(self.pi)
        # L = len(Osequence)
        # path = [0] * L
        # delta = np.zeros([S, L])
        # b_delta = np.zeros([S, L])
        #
        # output = self.obs_dict[Osequence[0]]
        # for i in range(S):
        #     delta[i][0] = self.pi[i] * self.B[i][output]
        #
        # for i in range(1, L):
        #     output = self.obs_dict[Osequence[i]]
        #     for j in range(S):
        #         maximum = 0
        #         arg_max = 0
        #         for sp in range(S):
        #             ad = self.A[sp][j] * delta[sp][i - 1]
        #             if maximum < ad:
        #                 maximum = ad
        #                 arg_max = sp
        #         delta[j][i] = self.B[j][output] * maximum
        #         b_delta[j][i] = arg_max
        #
        # maximum = 0
        # arg_max = 0
        # for i in range(S):
        #     if maximum < delta[i][L - 1]:
        #         maximum = delta[i][L - 1]
        #         arg_max = i
        # path[L - 1] = arg_max
        #
        # # Recursion
        # for i in range(0, L - 1):
        #     t = L - 2 - i
        #     path[t] = int(b_delta[int(path[t + 1])][t + 1])
        #
        # # States from iex
        # key_list = list(self.state_dict.keys())
        # val_list = list(self.state_dict.values())
        # for t in range(L):
        #     val = path[t]
        #     path[t] = key_list[val_list.iex(val)]

        S = len(self.pi)
        L = len(Osequence)
        temp = np.zeros((L, S))
        i = np.array([self.obs_dict[it] for it in Osequence])
        temp[0] = self.pi * self.B[:, i[0]]

        for j in range(1, L):
            temp[j] = self.B[:, i[j]] * np.max(np.multiply(self.A, np.expand_dims(temp[j - 1], 1)), axis=0)

        path_i = np.zeros(L)
        dict_val = {v: k for k, v in self.state_dict.items()}
        path_i[L - 1] = np.argmax(temp[L - 1])

        for j in range(L - 2, -1, -1):
            path_i[j] = np.argmax(self.A[:, int(path_i[j + 1])] * temp[j])
        path = [dict_val[x] for x in path_i]
        ###################################################
        return path
