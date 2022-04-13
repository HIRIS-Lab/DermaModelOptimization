from unittest import result
from torch import Tensor, argmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class Metrics():
    @staticmethod
    def accuracy(pred: Tensor, targets: Tensor) -> float:
        r"""
            Accuracy classification score.
            Args:
                pred (Tensor): A tensor which contains the prediction of the binary classificator.
                    This tensor contains the information using one-hot encoding.
                    
                targets (Tensor): A tensor which contains the real classification of the samples.
                    This tensor does not use one-hot encoding, classes are enumerated by unsigned
                    integer values.
            Returns:
                Accuracy value.
        """
        return accuracy_score(targets.cpu(), argmax(pred, dim=1).cpu(), normalize=True)

    @staticmethod
    def performance(pred: Tensor, targets: Tensor, weights=None) -> tuple:
        r"""
            Obtain statistical measures of the performance of a binary classification test
            that are widely used: sensitivity, specificity and precission.
            Args:
                pred (Tensor): A tensor which contains the prediction of the binary classificator.
                    This tensor contains the information using one-hot encoding.
                targets (Tensor): A tensor which contains the real classification of the samples.
                    THis tensor does not use one-hot encoding, classes are enumerated by unsigned
                    integer values.
            Return:
                Return a tuple with contains the following metrics: sensitivity, specificity
                and precission.
        """
        tn, fp, fn, tp = confusion_matrix(targets.cpu(), argmax(pred, dim=1).cpu(),labels=[0,1],sample_weight=weights).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (tp+fn)
        precission = tp / (tp+fp)
        recall = tp / (tp+fn)

        return (sensitivity, specificity, precission, recall)
