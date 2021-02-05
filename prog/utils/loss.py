import numpy as np


def softmax_ce_naive_forward_backward(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) + une régularisation L2 et le gradient des poids. Utilise une 
       activation softmax en sortie.
       
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2
       
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemple d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """

    N = X.shape[0]
    C = W.shape[1]

    loss = 0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #
    # Iteration sur chaque vecteur x_i de la batch X
    for i in range(N):
        # Calcul du score du x_i
        score = np.dot(X[i], W)
        # Exponentiel du score
        exp_score = np.exp(score)
        # Normalisation afin d'obtenir les P(C_i|x_i)
        s = exp_score / np.sum(exp_score)
        loss += - np.log(s[y[i]])
        for j  in range(C):
            if j == y[i]:
                dW[:, j] +=  X[i] * (s[j] - 1)
            else:
                dW[:, j] += X[i] * (s[j])
    # Moyenne et regularisation
    loss = loss / N + 0.5 * reg * np.linalg.norm(W) ** 2
    dW = dW / N + reg * W

    return loss, dW


def softmax_ce_forward_backward(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) et le gradient des poids. Utilise une activation softmax en sortie.
        
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2      
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemples d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    N = X.shape[0]
    C = W.shape[1]
    loss = 0.0
    dW = np.zeros(W.shape)
    ### TODO ###
    # Ajouter code ici #
    # # Transformation de y en one hot vectors
    y_one_hot = np.zeros((N, C))
    y_one_hot[np.arange(N), y] = 1
    # Calcul du score pour le batch X
    score = np.matmul(X, W)
    # Normalisation du score pour eviter explosion de la fonction exp
    # https://cs231n.github.io/linear-classify/#softmax
    score -= np.matrix(np.max(score, axis=1)).T
    exp_score = np.exp(score)
    # Pour éviter les erreurs de divsion resultante à des petites valeurs,
    # Ecrire la loss = -score_t + log(\sum(e^{score_j})) par propriété de la
    # fonction log : https://cs231n.github.io/linear-classify/#softmax
    sumline_exp_score = np.sum(exp_score, axis=1)
    losses = -score[np.arange(N), y]
    losses += np.log(sumline_exp_score)
    # Application de la softmax
    softmax_out = exp_score / np.matrix(sumline_exp_score).T

    # Calcul de la moyennne et ajout de la regularisation
    loss = np.sum(losses) / N
    loss += 0.5 * reg * np.linalg.norm(W) ** 2
    dW = np.matmul(X.T, (softmax_out - y_one_hot)) / N
    dW += reg * W

    return loss, dW


def hinge_naive_forward_backward(X, W, y, reg):
    """Implémentation naive calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.
       
       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #
    N = X.shape[0]
    for index,data in enumerate(X):
        forward = np.dot(data,W)
        pred = np.argmax(forward)
        pred_score = forward[pred]
        real_score = forward[y[index]]
        loss += np.max([0.0 ,1.0 + pred_score - real_score])
        if pred!=y[index]:
            dW[:,pred] += data
            dW[:,y[index]] -= data
    # Calcul de la moyenne plus la regularisation
    dW /= N
    dW += reg * W
    loss /= N
    loss += 0.5 * reg * np.linalg.norm(W) ** 2

    return loss, dW


def hinge_forward_backward(X, W, y, reg):
    """Implémentation vectorisée calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.

       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!
       
    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #
    forward = np.matmul(X, W)
    real_score = forward[np.arange(forward.shape[0]), y]
    diff_to_real = forward - np.matrix(real_score).T
    loss_vector = np.max(diff_to_real, axis=1) + 1
    loss = np.mean(loss_vector) + 0.5 * reg * np.linalg.norm(W) ** 2

    pred_index = np.argmax(forward, axis=1)
    grad_matrix = np.zeros_like(forward)

    grad_matrix[np.arange(grad_matrix.shape[0]), pred_index] = 1
    grad_matrix[np.arange(grad_matrix.shape[0]), y] -= 1
    dW = 1/X.shape[0] * np.matmul(X.T, grad_matrix) + reg * W

    return loss, dW
