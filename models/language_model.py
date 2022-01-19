from transformers import BertTokenizer, BertModel
import torch


class Language_Model(torch.nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained("bert-base-uncased")
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        return output


