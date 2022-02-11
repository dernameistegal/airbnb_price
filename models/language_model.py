from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F


class LanguageBertNet(nn.Module):
    def __init__(self, fine_tune=False, with_attention_masks=True, cls_dropout_prob=0, modelstring="bert-base-uncased"):
        super(LanguageBertNet, self).__init__()

        self.with_attention_masks = with_attention_masks

        self.bert = BertModel.from_pretrained(modelstring, output_attentions=True)

        # Turn gradients for BertModel on/off
        self.bert.requires_grad_(fine_tune)

        # model for embeddings
        self.bn1 = nn.BatchNorm1d(768)
        self.bn2 = nn.BatchNorm1d(100)

        self.drop1 = nn.Dropout(cls_dropout_prob)
        self.drop2 = nn.Dropout(0.5)

        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, input_ids, attention_mask=None):
        if not self.with_attention_masks:
            attention_mask = None
        # Bert result
        result = self.bert(input_ids, attention_mask)
        attentions = result["attentions"]

        # embeddings from Bert result
        embedding = result["last_hidden_state"][:, 0, :]
        embedding = self.bn1(embedding)
        embedding = self.drop1(embedding)

        # processing of embeddings
        x = self.bn2(F.relu(self.linear1(embedding)))
        x = self.drop2(x)
        x = self.linear2(x)

        return x, attentions

    def genembeddings(self, input_ids, attention_mask=None):
        if not self.with_attention_masks:
            attention_mask = None
        # Bert result
        result = self.bert(input_ids, attention_mask)
        attentions = result["attentions"]

        # embeddings from Bert result
        embedding = result["last_hidden_state"][:, 0, :]
        embedding = self.bn1(embedding)
        embedding = self.drop1(embedding)
        x = self.bn2(self.linear1(embedding))

        return x
