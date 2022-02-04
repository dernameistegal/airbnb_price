from captum.attr import visualization
import torch
from transformers import BertTokenizer


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


def visualize_attention(text, attention):
    for subatt in attention:
        subatt = subatt[0, 0, :, :][:, 0]
        vis_data_records = [visualization.VisualizationDataRecord(subatt, 0, 0, 0, 0, 0, text.split(), 1)]
        visualization.visualize_text(vis_data_records)

