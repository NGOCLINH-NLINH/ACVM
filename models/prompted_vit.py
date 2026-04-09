import torch
import torch.nn as nn
import timm


class PromptedViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', prompt_length=10):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.embed_dim
        self.prompt_length = prompt_length

        self.visual_prompts = nn.Parameter(torch.zeros(1, prompt_length, self.embed_dim))

        nn.init.trunc_normal_(self.visual_prompts, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        prompts = self.visual_prompts.expand(B, -1, -1)
        x = torch.cat((x[:, :1, :], prompts, x[:, 1:, :]), dim=1)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        z = F.normalize(x[:, 0], p=2, dim=1)

        return z


