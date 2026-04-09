import torch
import torch.nn as nn
import torch.nn.functional as F


class ACVMLoss(nn.Module):
    def __init__(self, alpha_base=0.5, delta=0.5, beta=1.0, lambda_spread=1.0):
        super().__init__()
        self.alpha_base = alpha_base
        self.delta = delta
        self.beta = beta
        self.lambda_spread = lambda_spread

    def compute_adaptive_triplet_loss(self, z, target_anchors, all_anchors, labels):
        B = z.size(0)
        C = all_anchors.size(0)

        sim_pos = F.cosine_similarity(z, target_anchors)

        sim_all = F.cosine_similarity(z.unsqueeze(1), all_anchors.unsqueeze(0), dim=-1)

        anchor_sim = F.cosine_similarity(target_anchors.unsqueeze(1), all_anchors.unsqueeze(0), dim=-1)
        adaptive_margin = self.alpha_base * (1.0 - anchor_sim)

        loss_matrix = F.relu(sim_all - sim_pos.unsqueeze(1) + adaptive_margin)
        mask = torch.ones_like(loss_matrix, dtype=torch.bool)
        mask[torch.arange(B), labels] = False

        loss_m = loss_matrix[mask].mean()

        return loss_m

    def compute_spread_regularization(self, z, all_anchors, labels):
        B = z.size(0)
        sim_all = F.cosine_similarity(z.unsqueeze(1), all_anchors.unsqueeze(0), dim=-1)

        mask = torch.ones_like(sim_all, dtype=torch.bool)
        mask[torch.arange(B), labels] = False

        sim_negative = sim_all[mask]

        loss_spread = F.relu(sim_negative - self.delta).sum() / B

        return loss_spread

    def compute_semantic_distance_loss(self, z_current, z_old, past_anchors):
        dist_current = 1.0 - F.cosine_similarity(z_current.unsqueeze(1), past_anchors.unsqueeze(0), dim=-1)
        dist_old = 1.0 - F.cosine_similarity(z_old.unsqueeze(1), past_anchors.unsqueeze(0), dim=-1)
        loss_d = F.mse_loss(dist_current, dist_old)

        return loss_d

    def forward(self, z, target_anchors, all_anchors, labels):
        loss_m = self.compute_adaptive_triplet_loss(z, target_anchors, all_anchors, labels)
        loss_spread = self.compute_spread_regularization(z, all_anchors, labels)

        total_loss = loss_m + self.lambda_spread * loss_spread

        return total_loss, loss_m, loss_spread
