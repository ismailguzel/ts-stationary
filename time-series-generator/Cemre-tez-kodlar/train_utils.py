from tqdm import tqdm
import torch
import numpy as np

def prfa(preds, labels):

    # Flatten inputs to handle the batch and prediction dimensions together
    preds = preds.flatten()
    labels = labels.flatten()
    
    # True Positives, False Positives, False Negatives, and True Negatives
    tp = np.sum((preds == 1) & (labels == 1)).astype(np.float32)
    fp = np.sum((preds == 1) & (labels == 0)).astype(np.float32)
    fn = np.sum((preds == 0) & (labels == 1)).astype(np.float32)
    tn = np.sum((preds == 0) & (labels == 0)).astype(np.float32)

    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    # Accuracy: (TP + TN) / Total
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    f1_score *= 100.0
    accuracy *= 100.0
    precision *= 100.0
    recall *= 100.0
    
    return np.float32(precision), np.float32(recall), np.float32(f1_score), np.float32(accuracy)

    
def calculate_metrics(model, loaders, device):
    model.eval()
    
    mean_all_preds = []
    mean_all_labels = []
    var_all_preds = []
    var_all_labels = []
    trend_all_preds = []
    trend_all_labels = []
    anomaly_all_preds = []
    anomaly_all_labels = []
    assert isinstance(loaders, list)

    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                data, mean_shift, variance_shift, trend_shift, anomaly = batch
                
                data = data.to(device)
                mean_shift = mean_shift.to(device)
                variance_shift = variance_shift.to(device)
                trend_shift = trend_shift.to(device)
                anomaly = anomaly.to(device)
          
                
                output = model(data)
                
                predicted = (output > 0.0)
                mean_shift_pred = predicted[:, 0, :]
                variance_shift_pred = predicted[:, 1, :]
                trend_shift_pred = predicted[:, 2, :]
                anomaly_pred = predicted[:, 3, :]
    
                mean_all_preds.append(mean_shift_pred.cpu().numpy())
                var_all_preds.append(variance_shift_pred.cpu().numpy())
                trend_all_preds.append(trend_shift_pred.cpu().numpy())
                anomaly_all_preds.append(anomaly_pred.cpu().numpy())
                mean_all_labels.append(mean_shift.cpu().numpy())
                var_all_labels.append(variance_shift.cpu().numpy())
                trend_all_labels.append(trend_shift.cpu().numpy())
                anomaly_all_labels.append(anomaly.cpu().numpy())

    #breakpoint()
    mean_all_preds = np.concatenate(mean_all_preds, axis = 1).flatten()
    mean_all_labels = np.concatenate(mean_all_labels, axis = 1).flatten().astype(np.bool)
    var_all_preds = np.concatenate(var_all_preds, axis = 1).flatten()
    var_all_labels = np.concatenate(var_all_labels, axis = 1).flatten().astype(np.bool)
    trend_all_preds = np.concatenate(trend_all_preds, axis = 1).flatten()
    trend_all_labels = np.concatenate(trend_all_labels, axis = 1).flatten().astype(np.bool)
    anomaly_all_preds = np.concatenate(anomaly_all_preds, axis = 1).flatten()
    anomaly_all_labels = np.concatenate(anomaly_all_labels, axis = 1).flatten().astype(np.bool)
    # print('mean')
    # print(mean_all_labels.min(), mean_all_labels.max())
    # print(mean_all_preds.min(), mean_all_preds.max())
    # print('var')
    # print(var_all_labels.min(), var_all_labels.max())
    # print(var_all_preds.min(), var_all_preds.max())
    # print('trend')
    # print(trend_all_labels.min(), trend_all_labels.max())
    # print(trend_all_preds.min(), trend_all_preds.max())
    # print('anomaly')
    # print(anomaly_all_labels.min(), anomaly_all_labels.max())
    # print(anomaly_all_preds.min(), anomaly_all_preds.max())

    #print('calculating mean')
    mean_shift_prec, mean_shift_recall, mean_shift_f1, mean_shift_acc = prfa(mean_all_labels, mean_all_preds)
    #print('calculating var')
    var_shift_prec, var_shift_recall, var_shift_f1, var_shift_acc = prfa(var_all_labels, var_all_preds)
    #print('calculating tred')
    trend_shift_prec, trend_shift_recall, trend_shift_f1, trend_shift_acc = prfa(trend_all_labels, trend_all_preds)
    #print('calculating ano')
    anomaly_prec, anomaly_recall, anomaly_f1, anomaly_acc = prfa(anomaly_all_labels, anomaly_all_preds)
    

    model.train()
    return_dict = {
        "mean_shift_accuracy": mean_shift_acc,
        "mean_shift_precision": mean_shift_prec,
        "mean_shift_recall": mean_shift_recall,
        "mean_shift_f1": mean_shift_f1,
        "variance_shift_accuracy": var_shift_acc,
        "variance_shift_precision": var_shift_prec,
        "variance_shift_recall": var_shift_recall,
        "variance_shift_f1": var_shift_f1,
        "trend_shift_accuracy": trend_shift_acc,
        "trend_shift_precision": trend_shift_prec,
        "trend_shift_recall": trend_shift_recall,
        "trend_shift_f1": trend_shift_f1,
        "anomaly_accuracy": anomaly_acc,
        "anomaly_precision": anomaly_prec,
        "anomaly_recall": anomaly_recall,
        "anomaly_f1": anomaly_f1,
        "avg_f1": np.mean([mean_shift_f1, var_shift_f1, trend_shift_f1, anomaly_f1], dtype=np.float32),
        "avg_precision": np.mean([mean_shift_prec, var_shift_prec, trend_shift_prec, anomaly_prec], dtype=np.float32),
        "avg_recall": np.mean([mean_shift_recall, var_shift_recall, trend_shift_recall, anomaly_recall], dtype=np.float32),
        "avg_accuracy": np.mean([mean_shift_acc, var_shift_acc, trend_shift_acc, anomaly_acc], dtype=np.float32),
    }

    #print(return_dict)
    return return_dict










    