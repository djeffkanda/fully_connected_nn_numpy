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
    for index,data in enumerate(X):
        forward = np.dot(data,W)
        pred = np.argmax(forward)
        pred_score = forward[pred]
        real_score = forward[y[index]]
        loss += np.max([0.0 ,1.0 + pred_score - real_score]) + 0.5*reg*np.linalg.norm(W)**2
        if pred!=y[index]:
            dW[:,pred] += 1/X.shape[0]*data
            dW[:,y[index]] -= 1/X.shape[0]*data
        dW += reg*W

    loss = loss/X.shape[0]

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

    return loss, dW
