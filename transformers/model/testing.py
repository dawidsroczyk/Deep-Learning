import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay


def test_model(model, testloader, device, folder, target_names):

    os.makedirs(folder, exist_ok=True)

    ids = []
    predicted_labels = []
    actual_labels = []
    success_indicators = []

    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            batch_size = images.size(0)
            ids.extend(range(i * batch_size + 1, (i + 1) * batch_size + 1))
            predicted_labels.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            success_indicators.extend((predicted == labels).cpu().numpy().astype(int))

    results_df = pd.DataFrame({
        'id': ids,
        'predicted_label': predicted_labels,
        'actual_label': actual_labels,
        'success_indicator': success_indicators
    })

    results_file = os.path.join(folder, 'tests.csv')
    results_df.to_csv(results_file, index=False)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    conf_matrix = confusion_matrix(actual_labels, predicted_labels)
    class_report = classification_report(actual_labels, predicted_labels, target_names=target_names)

    print(f"Accuracy: {accuracy:.4f}")
    #print("\nClassification Report:")
    #print(class_report)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=target_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(folder, 'confusion_matrix.pdf'), format='pdf')
    #plt.show()

    normalized_cm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=normalized_cm,
                                display_labels=target_names)
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title("Normalized Confusion Matrix")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(folder, 'confusion_matrix_norm.pdf'), format='pdf')
    #plt.show()

    analysis_file = os.path.join(folder, 'analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report)
        f.write("\n\nConfusion Matrix:\n")
        np.savetxt(f, conf_matrix, fmt='%d')  # Write integer matrix
        f.write("\n\nNormalized Confusion Matrix (%):\n")
        np.savetxt(f, normalized_cm, fmt='%.1f')  # Write float matrix with 1 decimal
        
    return accuracy