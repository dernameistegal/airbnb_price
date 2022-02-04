from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


class LanguageBertNet(nn.Module):
    def __init__(self, fine_tune=False, with_attention_masks=True, cls_dropout_prob=0, modelstring="bert-base-uncased"):
        super(LanguageBertNet, self).__init__()

        self.with_attention_masks = with_attention_masks

        # TODO: BertModel as an embedding layer
        self.bert = BertModel.from_pretrained(modelstring, output_attentions=True)

        # Turn gradients for BertModel on/off
        # (Fine-tuning is an optional task at the end of this exercise sheet)
        self.bert.requires_grad_(fine_tune)

        # TODO: Classification layer
        self.linear1 = nn.Linear(768, 768)
        self.linear2 = nn.Linear(768, 1)
        self.drop1 = nn.Dropout(cls_dropout_prob)

    def forward(self, input_ids, attention_mask=None):
        if not self.with_attention_masks:
            attention_mask = None
        result = self.bert(input_ids, attention_mask)
        embedding = result["last_hidden_state"][:, 0, :]
        embedding = self.drop1(embedding)
        attentions = result["attentions"]
        x = F.relu(self.linear1(embedding))
        x = self.linear2(x)
        return x, attentions
