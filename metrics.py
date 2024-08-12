import json
from matplotlib import pyplot as plt
from sklearn.base import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
import seaborn as sns

def calculate_metrics(predictions):
    logits = predictions.output.prediction
    true_labels = predictions.x['decoder_target']

    logits_cpu = logits.cpu()
    true_labels_cpu = true_labels.cpu()

    probabilities = F.softmax(logits_cpu.clone().detach(), dim=-1)
    probabilities_avg = probabilities.mean(dim=1)
    print(f"Shape of probabilities: {probabilities.shape}")

    class_predictions = np.argmax(probabilities, axis=-1)

    class_predictions_flat = class_predictions.view(-1).numpy()
    true_labels_flat = true_labels_cpu.view(-1).numpy()
    
    train_accuracy = accuracy_score(true_labels_flat, class_predictions_flat)

    print(classification_report(true_labels_flat, class_predictions_flat, zero_division=0))

    precision = precision_score(true_labels_flat, class_predictions_flat, average='weighted')
    recall = recall_score(true_labels_flat, class_predictions_flat, average='weighted')
    f1 = f1_score(true_labels_flat, class_predictions_flat, average='weighted')

    conf_matrix = confusion_matrix(true_labels_flat, class_predictions_flat)

    roc_auc = roc_auc_score(true_labels_flat, probabilities_avg, multi_class='ovr')
    log_loss_value = log_loss(true_labels_flat, probabilities_avg)

    metrics = {
    "accuracy": train_accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": conf_matrix.tolist(),
    "roc_auc": roc_auc,
    "log_loss": log_loss_value
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Die Metriken wurden in metrics.json gespeichert.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Validation')
    plt.savefig('conf_matrix.png')
    plt.show()