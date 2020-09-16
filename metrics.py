import torch
import torch.nn.functional as F


# SR : Segmentation Result
# GT : Ground Truth

def confusion(prediction, truth, threshold=0.5):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """

    prediction = (prediction > threshold).type(torch.uint8)
    confusion_vector = prediction / truth
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives


def get_accuracy(SR, GT, threshold=0.5):
    TP, FP, TN, FN = confusion(SR, GT)
    accuracy = float(TP + TN) / (float(TP + TN + FP + FN) + 1e-6)

    return accuracy


def get_sensitivity(SR, GT, threshold=0.5):

    TP, FP, TN, FN = confusion(SR, GT)

    SE = float(TP) / (float(TP + FN) + 1e-6)

    return SE


def get_specificity(SR, GT, threshold=0.5):

    TP, FP, TN, FN = confusion(SR, GT)

    SP = float(TN) / (float(TN + FP) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    TP, FP, TN, FN = confusion(SR, GT)
    PC = float(TP) / (float(TP + FP) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=0.5):
    SR = (SR > threshold).type(torch.uint8)
    Inter = torch.sum(((SR + GT) == 2).type(torch.uint8))
    Union = torch.sum(((SR + GT) >= 1).type(torch.uint8))

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).type(torch.uint8)

    Inter = torch.sum(((SR + GT) == 2).type(torch.uint8))
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)

    return DC
