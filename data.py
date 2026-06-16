import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizerFast


class SpamDataset(Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
            "labels": self.labels[idx],
        }


def create_dataloader(batch_size, max_len=128, val_split=0.1, test_split=0.1):
    dataset = load_dataset("sms_spam", split="train")

    texts = dataset["sms"]
    labels = dataset["label"]   # 0 = ham, 1 = spam

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    token_type_ids = encodings["token_type_ids"]
    labels = torch.tensor(labels, dtype=torch.long)

    n = len(labels)
    indices = torch.randperm(n)

    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    def make_dataset(idx):
        return SpamDataset(
            input_ids[idx],
            attention_mask[idx],
            token_type_ids[idx],
            labels[idx],
        )

    train_loader = DataLoader(make_dataset(train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(make_dataset(val_idx), batch_size=batch_size)
    test_loader = DataLoader(make_dataset(test_idx), batch_size=batch_size)

    vocab_size = tokenizer.vocab_size   # 30522 for bert-base-uncased

    return train_loader, val_loader, test_loader, vocab_size