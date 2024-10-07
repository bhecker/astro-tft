import json
from matplotlib import pyplot as plt
import numpy as np
from pytorch_forecasting import NaNLabelEncoder, TemporalFusionTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score, accuracy_score
import seaborn as sns

def calculate_metrics():
    model = TemporalFusionTransformer.load_from_checkpoint("best-checkpoint-20240918-165encoder.ckpt")
    # if hasattr(model, 'output_transformer'):
    #     output_transformer = model.output_transformer
    #     class_names = output_transformer.classes_

    class_names = ['SNIa', 'Dwarf Novae', 'Microlenses', 'Cepheids']

    class_predictions_flat = np.load(f'predictions-test/final_combined_class_predictions.npy')
    true_labels_flat = np.load(f'predictions-test/final_combined_true_labels.npy')
    probabilities_avg = np.load(f'predictions-test/final_combined_probabilities_avg.npy')
    
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

    print("Die Metriken wurden in metrics.json gespeichert.")

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