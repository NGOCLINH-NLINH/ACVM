import torch
from torch.optim import Adam
import time

import argparse
import copy
from core import ACVMLoss
from datasets.seq_cifar100 import build_task_loaders
from models import PromptedViT, DynamicSemanticAnchor
from utils import ContinualMetrics
from utils.metrics import evaluate_seen_classes


def parse_args():
    parser = argparse.ArgumentParser(description="Training ACVM")

    parser.add_argument('--epochs', type=int, default=20, help='Epochs per task')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--prompt_length', type=int, default=10, help='Visual Prompt length')
    parser.add_argument('--ctx_length', type=int, default=8, help='Text Context length')
    parser.add_argument('--alpha_base', type=float, default=0.5, help='Margin for Triplet Loss')
    parser.add_argument('--delta', type=float, default=0.5, help='Margin for Spread Loss')
    parser.add_argument('--lambda_spread', type=float, default=1.0, help='Spread Loss weight')
    parser.add_argument('--lambda_kd', type=float, default=5.0, help='KD Loss weight')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    print("[INFO] Initializing PromptedViT and DynamicSemanticAnchor")
    visual_encoder = PromptedViT(model_name='vit_small_patch16_224', prompt_length=10).to(device)
    anchor_generator = (
        DynamicSemanticAnchor(model_name='sentence-transformers/all-MiniLM-L6-v2', ctx_length=args.ctx_length)
        .to(device))

    criterion = ACVMLoss(alpha_base=args.alpha_base, delta=args.delta, lambda_spread=args.lambda_spread).to(device)
    num_tasks = 10
    epochs_per_task = args.epochs
    metrics = ContinualMetrics(num_tasks=num_tasks)

    optimizer = Adam([
        {'params': visual_encoder.visual_prompts, 'lr': args.lr},
        {'params': anchor_generator.ctx, 'lr': args.lr}
    ], weight_decay=1e-4)

    print("[INFO] Preparing dataset Seq-CIFAR100")
    task_loaders, task_classes_list = build_task_loaders(data_dir='./data', num_tasks=num_tasks,
                                                         batch_size=args.batch_size)

    seen_classes_str = []
    seen_classes_idx = []

    prev_visual_encoder = None

    global_frozen_anchors = []

    for task_id in range(num_tasks):
        print(f"\n" + "=" * 50)
        print(f" START TRAINING TASK {task_id + 1}/{num_tasks} ")
        print(f" Classes: {task_classes_list[task_id]}")
        print("=" * 50)

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
            epoch_kd = 0.0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                z = visual_encoder(images)

                new_anchors = torch.stack([anchor_generator(cls_name) for cls_name in current_classes_str])

                if len(global_frozen_anchors) > 0:
                    past_anchors = torch.stack(global_frozen_anchors).to(device)
                    all_anchors = torch.cat([past_anchors, new_anchors])
                else:
                    all_anchors = new_anchors

                target_anchors = all_anchors[labels]

                optimizer.zero_grad()
                total_loss, loss_m, loss_spread = criterion(z, target_anchors, all_anchors, labels)

                if prev_visual_encoder is not None:
                    with torch.no_grad():
                        z_old = prev_visual_encoder(images).detach()
                        past_anchors = torch.stack(global_frozen_anchors).to(device)  # <--- Lấy thẳng từ kho

                    loss_kd = criterion.compute_semantic_distance_loss(z, z_old, past_anchors)
                    total_loss = total_loss + args.lambda_kd * loss_kd
                    epoch_kd += loss_kd.item()

                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                epoch_triplet += loss_m.item()
                epoch_spread += loss_spread.item()

            num_batches = len(train_loader)
            if prev_visual_encoder is not None:
                print(f"Epoch [{epoch + 1}/{epochs_per_task}] | Loss: {epoch_loss / num_batches:.4f} "
                      f"(Triplet: {epoch_triplet / num_batches:.4f}, Spread: {epoch_spread / num_batches:.4f}, KD: {epoch_kd / num_batches:.4f})")
            else:
                print(f"Epoch [{epoch + 1}/{epochs_per_task}] | Loss: {epoch_loss / num_batches:.4f} "
                      f"(Triplet: {epoch_triplet / num_batches:.4f}, Spread: {epoch_spread / num_batches:.4f})")

        print(f"[INFO] Training task {task_id + 1} completed in {time.time() - start_time:.1f} seconds")
        anchor_generator.eval()
        with torch.no_grad():
            for cls_name in current_classes_str:
                frozen_anchor = anchor_generator(cls_name).detach()
                global_frozen_anchors.append(frozen_anchor)

        all_anchors_eval = torch.stack(global_frozen_anchors).to(device)

        print(f"\n[EVAL] Evaluating performance after task {task_id + 1}")
        visual_encoder.eval()

        with torch.no_grad():
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

        prev_visual_encoder = copy.deepcopy(visual_encoder)
        prev_visual_encoder.eval()
        for param in prev_visual_encoder.parameters():
            param.requires_grad = False

    print("\n" + "=" * 50)
    print(" ALL DONE ")
    print("=" * 50)


if __name__ == "__main__":
    main()
