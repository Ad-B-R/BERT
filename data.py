import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast


TRAIN_SUBSET = 50_000


class NLIDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels         = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels":         self.labels[idx],
        }


def create_dataloader(batch_size, max_len=128):
    dataset   = load_dataset("stanfordnlp/snli")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def prepare(split, max_samples=None):
        data = dataset[split].filter(lambda x: x["label"] != -1)
        if max_samples is not None:
            data = data.select(range(min(max_samples, len(data))))

        encodings = tokenizer(
            data["premise"],
            data["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return NLIDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            encodings["token_type_ids"],
            torch.tensor(data["label"], dtype=torch.long),
        )

    train_loader = DataLoader(prepare("train",      max_samples=TRAIN_SUBSET), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(prepare("validation"),                            batch_size=batch_size)
    test_loader  = DataLoader(prepare("test"),                                  batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer.vocab_size