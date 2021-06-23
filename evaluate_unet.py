from pathlib import Path
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch
import visdom as vis
import itertools

def precision(y_true, y_pred):
    one = torch.ones_like(torch.Tensor(y_true))
    one = one.numpy()
    TP = (y_true * y_pred).sum()
    FP = ((one-y_true)*y_pred).sum()
    return (TP + 1e-15) / (TP + FP + 1e-15)

def general_precision(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return precision(y_true, y_pred)

def recall(y_true, y_pred):
    one = torch.ones_like(torch.Tensor(y_pred))
    one = one.numpy()
    TP = (y_true * y_pred).sum()
    FN = (y_true*(one - y_pred)).sum()
    return (TP + 1e-15) / (TP + FN + 1e-15)

def general_recall(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return recall(y_true, y_pred)

def dice(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)

def general_dice(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return dice(y_true, y_pred)

def jaccard(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)

def general_jaccard(y_true, y_pred):
    if y_true.sum() == 0:
        if y_pred.sum() == 0:
            return 1
        else:
            return 0

    return jaccard(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ground_truth_dir', type=str, default='E:\\medical datasets\\Pancreas-CT\\test\\masks', help='path where ground truth images are located')
    parser.add_argument('-pred_dir', type=str, default='E:\\medical datasets\\Pancreas-CT\\test\\segnet_lstm\\prediction_output',  help='path with predictions')
    parser.add_argument('-threshold', type=float, default=0.24,  help='crack threshold detection')
    args = parser.parse_args()

    result_precision = []
    result_recall = []
    result_f1 = []
    result_dice = []
    result_jaccard = []

    # boxplot_precision = []
    # boxplot_recall = []
    # boxplot_f1 = []
    # boxplot_dice = []
    # boxplot_jaccard = []
    #
    # boxplot = []

    paths = [path for path in  Path(args.ground_truth_dir).glob('*')]
    for file_name in tqdm(paths):
        y_true = (cv2.imread(str(file_name), 0) > 0).astype(np.uint8)

        pred_file_name = Path(args.pred_dir) / file_name.name
        if not pred_file_name.exists():
            print(f'missing prediction for file {file_name.name}')
            continue

        pred_image = (cv2.imread(str(pred_file_name), 0) > 255 * args.threshold).astype(np.uint8)
        y_pred = pred_image

        result_precision += [precision(y_true, y_pred)]
        result_recall += [recall(y_true, y_pred)]
        result_dice += [dice(y_true, y_pred)]
        result_jaccard += [jaccard(y_true, y_pred)]
        result_f1 += [2*precision(y_true, y_pred)*recall(y_true, y_pred)/(recall(y_true, y_pred)+precision(y_true, y_pred))]

    # map(boxplot_precision, itertools.zip(*result_precision))
    # map(boxplot_recall, itertools.izip(*result_recall))
    # map(boxplot_f1, itertools.izip(*result_f1))
    # map(boxplot_dice, itertools.izip(*result_dice))
    # map(boxplot_jaccard, itertools.izip(*result_jaccard))

    # result_precision = list(result_precision)
    # result_recall = list(result_recall)
    # result_f1 = list(result_f1)
    # result_jaccard = list(result_jaccard)
    # result_dice = list(result_dice)

    # boxplot_precision = [[r[col] for r in result_precision] for col in range(len(result_precision[0]))]
    # boxplot_recall = [[r[col] for r in result_recall] for col in range(len(result_recall[0]))]
    # boxplot_f1 = [[r[col] for r in result_f1] for col in range(len(result_f1[0]))]
    # boxplot_dice = [[r[col] for r in result_dice] for col in range(len(result_dice[0]))]
    # boxplot_jaccard = [[r[col] for r in result_jaccard] for col in range(len(result_jaccard[0]))]
    # boxplot = boxplot_precision + boxplot_recall + boxplot_f1 + boxplot_dice + boxplot_jaccard

    print('Precision = ', np.mean(result_precision), np.std(result_precision))
    print('recall = ', np.mean(result_recall), np.std(result_recall))
    print('f1 = ', 2*np.mean(result_precision)*np.mean(result_recall)/(np.mean(result_precision)+np.mean(result_recall)), 2*np.std(result_precision)*np.std(result_recall)/(np.std(result_precision)+np.std(result_recall)))
    print('Dice = ', np.mean(result_dice), np.std(result_dice))
    print('Jaccard = ', np.mean(result_jaccard), np.std(result_jaccard))

    # vis.boxplot(
    #     X=boxplot,
    #     opts=dict(legend=['Precision', 'Recall', 'F1', 'Dice', 'IoU'])
    # )