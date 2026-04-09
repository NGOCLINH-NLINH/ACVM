import torch
from torch.optim import Adam
import time

from core import ACVMLoss
from datasets.seq_cifar100 import build_task_loaders
from models import PromptedViT, DynamicSemanticAnchor
from utils import ContinualMetrics
from utils.metrics import evaluate_seen_classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Initializing PromptedViT and DynamicSemanticAnchor")
    visual_encoder = PromptedViT(model_name='vit_base_patch16_224', prompt_length=10).to(device)
    anchor_generator = (DynamicSemanticAnchor(model_name='sentence-transformers/all-MiniLM-L6-v2', ctx_length=8)
                        .to(device))

    criterion = ACVMLoss(alpha_base=0.5, delta=0.5, lambda_spread=1.0).to(device)
    num_tasks = 10
    epochs_per_task = 5
    metrics = ContinualMetrics(num_tasks=num_tasks)

    optimizer = Adam([
        {'params': visual_encoder.visual_prompts, 'lr': 0.01},
        {'params': anchor_generator.ctx, 'lr': 0.01}
    ], weight_decay=1e-4)

    print("[INFO] Preparing dataset Seq-CIFAR100")
    task_loaders, task_classes_list = build_task_loaders(data_dir='./data', num_tasks=num_tasks, batch_size=64)

    seen_classes_str = []
    seen_classes_idx = []

    for task_id in range(num_tasks):
        print(f"\n" + "=" * 40)
        print(f" START TRAINING TASK {task_id + 1}/{num_tasks} ")
        print(f" Classes: {task_classes_list[task_id]}")
        print("=" * 40)

        current_classes_str = task_classes_list[task_id]
        seen_classes_str.extend(current_classes_str)

        start_idx = task_id * len(current_classes_str)
        end_idx = start_idx + len(current_classes_str)
        current_indices = list(range(start_idx, end_idx))
        seen_classes_idx.extend(current_indices)

        train_loader = task_loaders[task_id]['train']

        start_time = time.time()
        for epoch in range(epochs_per_task):
            visual_encoder.train()
            anchor_generator.train()

            epoch_loss = 0.0
            epoch_triplet = 0.0
            epoch_spread = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                z = visual_encoder(images)

                all_anchors = torch.stack([anchor_generator(cls_name) for cls_name in seen_classes_str])

                target_anchors = all_anchors[labels]

                optimizer.zero_grad()
                total_loss, loss_m, loss_spread = criterion(z, target_anchors, all_anchors, labels)

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_triplet += loss_m.item()
                epoch_spread += loss_spread.item()

            num_batches = len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs_per_task}] | Loss: {epoch_loss / num_batches:.4f} "
                  f"(Triplet: {epoch_triplet / num_batches:.4f}, Spread: {epoch_spread / num_batches:.4f})")

        print(f"[INFO] Training task {task_id + 1} completed in {time.time() - start_time:.1f} seconds")

        print(f"\n[EVAL] Evaluating performance after task {task_id + 1}")
        visual_encoder.eval()
        anchor_generator.eval()

        with torch.no_grad():
            all_anchors_eval = torch.stack([anchor_generator(cls_name) for cls_name in seen_classes_str])

            for eval_task_id in range(task_id + 1):
                eval_loader = task_loaders[eval_task_id]['test']

                acc = evaluate_seen_classes(
                    model=visual_encoder,
                    test_loader=eval_loader,
                    seen_indices=seen_classes_idx,
                    all_anchors_tensor=all_anchors_eval,
                    device=device
                )

                metrics.update_accuracy(current_task=task_id, eval_task=eval_task_id, accuracy=acc)
                print(f"  -> Accuracy on task {eval_task_id + 1}: {acc:.2%}")

        avg_acc = metrics.get_average_accuracy(task_id)
        forgetting = metrics.get_forgetting(task_id)
        print(f"\n>>> OVERALL RESULT AFTER TASK {task_id + 1}:")
        print(f"    Average Accuracy (A_{task_id + 1}): {avg_acc:.2%}")
        print(f"    Forgetting: {forgetting:.2%}")

    print("\n" + "=" * 50)
    print(" ALL DONE ")
    print("=" * 50)


if __name__ == "__main__":
    main()
