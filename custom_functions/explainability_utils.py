from captum.attr import visualization
import torch
from transformers import BertTokenizer
import numpy as np


def pipeline(text, model, device, max_len=128, modelstring="bert-base-uncased"):
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(modelstring, do_lower_case=True)

    sequence = tokenizer.encode_plus(
        text,  # Review to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        truncation=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True)
    tokens = torch.tensor(sequence['input_ids']).to(device).reshape(1, max_len)
    attention_mask = torch.tensor(sequence['attention_mask']).to(device).reshape(1, max_len)
    with torch.no_grad():
        output = model(tokens, attention_mask)
        price = torch.exp(output[0]).item()
        attention = output[1]
    print(price)
    return attention


def visualize_attention(text, attention, offset=0):
    ls = []
    for subatt in attention:
        subatt = subatt[0, 0, :, :][:, 0].cpu().numpy()
        ls.append(subatt / np.max(subatt))
    ls = np.array(ls)
    ls = np.sum(ls, axis=0)
    ls = ls - offset
    vis_data_records = [visualization.VisualizationDataRecord(ls, 0, 0, 0, 0, 0, text.split(), 1)]
    visualization.visualize_text(vis_data_records)

