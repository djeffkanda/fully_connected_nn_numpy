import numpy as np


def cross_entropy_loss(scores, t, reg, model_params):
    """Calcule l'entropie croisée multi-classe.

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régularisation
        model_params {dict} -- Dictionnaire contenant les paramètres de chaque couche
                               du modèle. Voir la méthode "parameters()" de la
                               classe Model.

    Returns:
        loss {float} 
        dScore {ndarray}: dérivée de la loss par rapport aux scores. : Shape (N, C)
        softmax_output {ndarray} : Shape (N, C)
    """
    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    softmax_output = np.zeros(scores.shape)
    
    # TODO
    # Ajouter code ici
    #Calcul du softmax
    exp_score = np.exp(scores)
    softmax_output = exp_score / np.matrix(np.sum(exp_score, axis=1)).T

    #Calcul de la loss
    t_one_hot = np.zeros((N, C))
    t_one_hot[np.arange(N), t] = 1
    losses = - np.log(softmax_output[np.arange(N), t])
    # Calcul de la moyennne et ajout de la regularisation
    R = 0
    for key in model_params.keys():
        R += np.linalg.norm(model_params[key]['W'])**2
    loss = np.mean(losses) + 0.5 * reg * R

    #Calcul du gradient
    dScores = softmax_output - t_one_hot

    return loss, dScores, softmax_output


def hinge_loss(scores, t, reg, model_params):
    """Calcule la loss avec la méthode "hinge loss multi-classe".

    Arguments:
        scores {ndarray} -- Scores du réseau (sortie de la dernière couche).
                            Shape (N, C)
        t {ndarray} -- Labels attendus pour nos échantillons d'entraînement.
                       Shape (N, )
        reg {float} -- Terme de régulatisation
        model_params {dict} -- Dictionnaire contenant l'ensemble des paramètres
                               du modèle. Obtenus avec la méthode parameters()
                               de Model.

    Returns:
        tuple -- Tuple contenant la loss et la dérivée de la loss par rapport
                 aux scores.
    """

    N = scores.shape[0]
    C = scores.shape[1]
    loss = 0
    dScores = np.zeros(scores.shape)
    score_correct_classes = np.zeros(scores.shape)

    # TODO
    # Ajouter code ici
    return loss, dScores, score_correct_classes
