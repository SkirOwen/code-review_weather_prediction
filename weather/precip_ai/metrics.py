from typing import Tuple
import torch


def wmse(y_actual: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Weighted Mean Square Error with weights
       embedded inside y_actual

    Args:
        y_actual: observed precipitation values- column 0, weights column 1
        y_pred: modeled precipitation values

    Returns:
        float: weighted mean square error
    """
    return torch.mean(y_actual[:, 1] * (y_actual[:, 0] - y_pred) ** 2)  # pylint: disable=no-member


@torch.jit.script
def contingency_table(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates a contingency (a.k.a. confusion matrix) table for a binary prediction.
    Args:
        y_true: actual observations
        y_pred: model prediction
        threshold: threshold to validate
    Returns:
        tp: true positive
        tn: true negative
        fp: false positive
        fn: false negative
    """
    y_true_cl = y_true.reshape([-1, 1]) > threshold
    y_pred_cl = y_pred.reshape([-1, 1]) > threshold
    tp = (y_pred_cl & y_true_cl).sum(1).float().sum()
    tn = ((~y_pred_cl) & (~y_true_cl)).sum(1).float().sum()
    fp = (y_pred_cl & (~y_true_cl)).sum(1).float().sum()
    fn = ((~y_pred_cl) & y_true_cl).sum(1).float().sum()
    return tp, tn, fp, fn


@torch.jit.script
def ets(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float) -> float:
    """
    Calculates the ETS score for a given threshold.
    See e.g. slide 19 of https://www.ecmwf.int/sites/default/files/elibrary/2007/15490-verification-categorical-forecasts.pdf
    Args:
        y_true: actual observations
        y_pred: model prediction
        threshold: the threshold to validate
    Returns:
        The ETS score.
    """
    tp, tn, fp, fn = contingency_table(y_true, y_pred, threshold)
    h_r = (tp + fp) * (tp + fn) / (tp + tn + fp + fn)
    sc = (tp - h_r) / (tp + fp + fn - h_r)
    return sc.item() if not torch.isnan(sc) else 0.0  # pylint: disable=no-member


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''
    FROM: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1