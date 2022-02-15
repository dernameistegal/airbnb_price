from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F


class LanguageBertNet(nn.Module):
    def __init__(self, fine_tune=False, with_attention_masks=True, cls_dropout_prob=0, modelstring="bert-base-uncased"):
        super(LanguageBertNet, self).__init__()

        self.with_attention_masks = with_attention_masks
        self.bert = BertModel.from_pretrained(modelstring, output_attentions=True)
        self.bert.requires_grad_(fine_tune)
        self.drop1 = nn.Dropout(cls_dropout_prob)

        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)

    def forward(self, input_ids, attention_mask=None):
        if not self.with_attention_masks:
            attention_mask = None
        result = self.bert(input_ids, attention_mask)
        attentions = result["attentions"]
        embedding = result["last_hidden_state"][:, 0, :]

        x = F.relu(self.linear1(embedding))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x, attentions

    def genembeddings(self, input_ids, attention_mask=None):
        if not self.with_attention_masks:
            attention_mask = None
        result = self.bert(input_ids, attention_mask)
        attentions = result["attentions"]
        embedding = result["last_hidden_state"][:, 0, :]
        x = self.drop1(embedding)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

    def prediction_from_mean(self, embedding):
        return self.linear2(embedding)
