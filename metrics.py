import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, accuracy_score
import seaborn as sns

def calculate_metrics():
    class_names = ['SNIa', 'Dwarf Novae', 'Microlenses', 'Cepheids']

    class_predictions_flat = np.load(f'predictions-test/final_combined_class_predictions.npy')
    true_labels_flat = np.load(f'predictions-test/final_combined_true_labels.npy')
    probabilities_avg = np.load(f'predictions-test/final_combined_probabilities_avg.npy')
    
    calculate_pr_curve(class_names, true_labels_flat, probabilities_avg)

    train_accuracy = accuracy_score(true_labels_flat, class_predictions_flat)
    print(classification_report(true_labels_flat, class_predictions_flat, target_names=class_names, zero_division=0))

    precision = precision_score(true_labels_flat, class_predictions_flat, average='weighted')
    recall = recall_score(true_labels_flat, class_predictions_flat, average='weighted')
    f1 = f1_score(true_labels_flat, class_predictions_flat, average='weighted')

    conf_matrix = confusion_matrix(true_labels_flat, class_predictions_flat, labels=list(range(len(class_names))))

    metrics = {
    "accuracy": train_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": conf_matrix.tolist(),
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved to metrics.json.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Validation')
    plt.savefig('conf_matrix.png')
    plt.show()


def calculate_pr_curve(class_names, true_labels_flat, probabilities_avg):
    true_labels_binarized = label_binarize(true_labels_flat, classes=[0, 1, 2, 3])

    plt.figure(figsize=(8, 6))

    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(true_labels_binarized[:, i], probabilities_avg[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{class_names[i]} (AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('pr_curve.png')
    plt.show()