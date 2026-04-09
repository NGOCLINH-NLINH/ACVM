import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


class DynamicSemanticAnchor(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', ctx_length=8):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_encoder = AutoModel.from_pretrained(model_name)

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.embed_dim = self.text_encoder.config.hidden_size

        self.ctx = nn.Parameter(torch.zeros(ctx_length, self.embed_dim))
        nn.init.trunc_normal_(self.ctx, std=0.02)

    def forward(self, class_name):
        tokens = self.tokenizer(class_name, return_tensors='pt', add_special_tokens=False)
        input_ids = tokens['input_ids'].to(self.ctx.device)
        with torch.no_grad():
            class_embeds = self.text_encoder.embeddings.word_embeddings(input_ids)
        ctx_expanded = self.ctx.unsqueeze(0)

        prompted_embeds = torch.cat([ctx_expanded, class_embeds], dim=1)

        outputs = self.text_encoder(inputs_embeds=prompted_embeds)

        raw_anchor = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        anchor = F.normalize(raw_anchor, p=2, dim=0)

        return anchor
