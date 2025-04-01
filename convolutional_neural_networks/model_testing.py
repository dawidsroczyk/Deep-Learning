import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import Subset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def test_model(model, testloader, device, folder):

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
    class_report = classification_report(actual_labels, predicted_labels, target_names=testloader.dataset.classes)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(class_report)

    analysis_file = os.path.join(folder, 'analysis.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("\n\nClassification Report:\n")
        f.write(class_report)


def plot_examples(model, testloader, device, class_names, folder, title, num_images=10):

    os.makedirs(folder, exist_ok=True)
    model.eval()

    test_dataset = testloader.dataset
    random_indices = np.random.choice(len(test_dataset), size=num_images, replace=False)
    random_subset = Subset(test_dataset, random_indices)
    random_loader = torch.utils.data.DataLoader(random_subset, batch_size=num_images, shuffle=False)

    images, labels = next(iter(random_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    images_np = images.cpu()
    actual_labels_np = labels.cpu().numpy()
    predicted_labels_np = predicted.cpu().numpy()

    plot_images_with_labels(
        images_np, actual_labels_np, predicted_labels_np,
        class_names=class_names,
        num_images=num_images,
        title=title,
        folder=folder
    )


def plot_images_with_labels(images, actual_labels, predicted_labels, class_names, num_images, title, folder):

    os.makedirs(folder, exist_ok=True)

    num_images = min(num_images, len(images))

    num_rows = (num_images + 4) // 5
    num_cols = min(num_images, 5)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    fig.suptitle(title, fontsize=16)

    axes = axes.flatten() if num_rows > 1 else axes

    for i in range(num_images):
        ax = axes[i]

        actual_label = class_names[actual_labels[i]]
        predicted_label = class_names[predicted_labels[i]]
        inp = torchvision.utils.make_grid(images[i])

        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.47889522, 0.47227842, 0.43047404])
        std = np.array([0.24205776, 0.23828046, 0.25874835])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        ax.imshow(inp)
        ax.set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}", fontsize=10)
        ax.axis('off')

    for j in range(num_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()

    plt.savefig(os.path.join(folder, 'example.pdf'), bbox_inches='tight')
    plt.show()