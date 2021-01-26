import numpy as np


class LinearClassifier:

    def __init__(self, forward_backward_function):
        self.W = None
        self.forward_backward_function = forward_backward_function

    def train(self, X, y, learning_rate=1e-3, weigth_scale=1e-3, reg=1e-5, num_iter=100, batch_size=200, verbose=False):
        """Boucle d'entraînement pour un classifieur linéaire.

        Arguments:
            X {ndarray} -- Données d'entraînement. Shape (N, D)
            y {ndarray} -- Labels d'entraînement. Shape (N, ).
                           Valeur y[i]: 0 <= y[i] < C, où C = nombre de classes

        Keyword Arguments:
            learning_rate {float} -- Taux d'apprentissage. (default: {1e-3})
            weigth_scale {float} -- Écart type de la distribution normale avec laquelle les poids
                                    sont initialisés. (default: {1e-3})
            reg {float} -- terme de régularisation. (default: {1e-5})
            num_iter {int} -- Nombre d'itérations de la boucle d'entraînement (default: {100})
            batch_size {int} -- Nombre de données d'entraînement passées au modèle par itération.
                                (default: {200})
            verbose {bool} -- (default: {False})

        Returns:
            list -- Historique de la perte lors de l'entraînement.
        """

        N, D = X.shape
        C = np.max(y) + 1

        if self.W is None:
            self.W = np.random.randn(D, C) * weigth_scale

        train_loss_history = []
        
        ### TODO ###
        # coder une boucle d'entrainement faisant "num_iter" iterations.
        for i in range(num_iter):
            # à chaque itération,
            #     faire une forward+backward pass afin de calculer la loss et le gradient pour une batch de "batch_size" elements choisis au hasard
            batch_choice = np.random.choice(N, batch_size, replace=False)
            X_batch = X[batch_choice]
            y_batch = y[batch_choice]
            loss, dW = self.forward_backward(X_batch, y_batch, reg)
            #     mettre effectuer une descente de gradient : W = W - learning_rate * gradient
            self.W -= learning_rate * dW
            #     ajoutér la loss de l'itération courante dans "train_loss_history"
            train_loss_history.append(loss)
            #     ajouter du code pour l'option *verbose*
            if verbose:
                print(
                    f'iteration : {i}, '
                    f'loss : {loss}'
                    , end='\r'
                )
        return train_loss_history

        return train_loss_history

    def predict(self, X):
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)

    def forward_backward(self, X, y, reg):
        loss, dW = self.forward_backward_function(X, self.W, y, reg)
        return loss, dW
