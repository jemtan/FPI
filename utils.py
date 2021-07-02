import os
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score


def save_roc_pr_curve_data(scores, labels, file_path=None):
    scores = scores.flatten()
    labels = labels.flatten()

    scores_pos = scores[labels == 1]
    scores_neg = scores[labels != 1]

    truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    ap = average_precision_score(truth, preds)

    if file_path:
        #save complete record
        np.savez_compressed(file_path,
            preds=preds, truth=truth,
            fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
            precision_norm=precision_norm, recall_norm=recall_norm,
            pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
            precision_anom=precision_anom, recall_anom=recall_anom,
            pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom,
            ap=ap)
    
    else:
        #just return scores
        score_dict = {'roc_auc':roc_auc,'pr_auc_norm':pr_auc_norm,
            'pr_auc_anom':pr_auc_anom,'ap':ap}
        return score_dict






