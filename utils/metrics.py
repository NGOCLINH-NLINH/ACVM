import numpy as np
import torch


class ContinualMetrics:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.R = np.zeros((num_tasks, num_tasks))
        self.linear_probe_acc = []
        self.zero_shot_acc = []

    def update_accuracy(self, current_task, eval_task, accuracy):
        self.R[current_task, eval_task] = accuracy

    def get_average_accuracy(self, current_task):
        return np.mean(self.R[current_task, :current_task + 1])

    def get_forgetting(self, current_task):
        if current_task == 0:
            return 0.0
        forgetting_scores = []
        for j in range(current_task):
            max_acc = np.max(self.R[:current_task, j])
            current_acc = self.R[current_task, j]
            forgetting_scores.append(max_acc - current_acc)
        return np.mean(forgetting_scores)


def evaluate_seen_classes(model, test_loader, seen_indices, all_anchors_tensor, device):
    model.eval()

    seen_indices_tensor = torch.tensor(seen_indices, dtype=torch.long, device=device)
    anchors_seen = all_anchors_tensor[seen_indices_tensor]

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            emb = model(images)

            sims = emb @ anchors_seen.t()

            local_preds = sims.argmax(dim=1)

            global_preds = seen_indices_tensor[local_preds]

            correct += (global_preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_zero_shot(model, test_loader, unseen_indices, all_anchors_tensor, device):
    if len(unseen_indices) == 0:
        return 0.0

    model.eval()
    unseen_indices_tensor = torch.tensor(unseen_indices, dtype=torch.long, device=device)
    anchors_unseen = all_anchors_tensor[unseen_indices_tensor]

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            emb = model(images)
            sims = emb @ anchors_unseen.t()

            local_preds = sims.argmax(dim=1)
            global_preds = unseen_indices_tensor[local_preds]

            correct += (global_preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0
