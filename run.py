from data import create_dataloader
import torch
import torch.nn as nn
import model
import json
import wandb


class BERT(nn.Module):
    def __init__(self, f, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.d_model  = f["model"]["d_model"]
        heads = f["model"]["heads"]
        dropout = f["model"]["dropout"]
        d_ff = f["model"]["d_ff"]
        n_layers = f["model"]["N"]
        max_len = f["model"]["max_len"]
        num_classes = f["model"]["num_classes"]

        embed = model.BERTEmbedding(
            vocab_size=vocab_size,
            dropout=dropout,
            max_len=max_len,
            d_model=self.d_model,
        )

        layers = nn.ModuleList([
            model.Encoder(
                features=self.d_model,
                dropout=dropout,
                attn=model.MultiHeadAttention(self.d_model, heads, dropout),
                ffn=model.FeedForwardNetwork(self.d_model, d_ff, dropout),
            )
            for _ in range(n_layers)
        ])

        self.transformer = model.BERT_Block(
            embed=embed,
            encoder=layers,
            d_model=self.d_model,
            num_classes=num_classes,
            dropout=dropout,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attn_mask=None):
        return self.transformer(input_ids, attn_mask)


with open("config.json") as f:
    config = json.load(f)


def train():
    wandb.init(
        project="bert-spam",
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, vocab_size = create_dataloader(
        batch_size=config["training"]["batch_size"],
        max_len=config["model"]["max_len"],
    )

    bert = BERT(config, vocab_size).to(device)
    wandb.watch(bert, log="gradients", log_freq=100)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        bert.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs        = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(epochs):

        bert.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch in train_loader:
            input_ids   = batch["input_ids"].to(device)
            attn_mask   = batch["attention_mask"].to(device)
            labels      = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = bert(input_ids, attn_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bert.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss    += loss.item() * input_ids.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total   += input_ids.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc   = train_correct / train_total

        bert.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)

                logits = bert(input_ids, attn_mask)
                loss   = criterion(logits, labels)

                val_loss    += loss.item() * input_ids.size(0)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total   += input_ids.size(0)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        wandb.log({
            "epoch":      epoch + 1,
            "train/loss": train_loss,
            "train/acc":  train_acc,
            "val/loss":   val_loss,
            "val/acc":    val_acc,
            "lr":         scheduler.get_last_lr()[0],
        })

        print(f"Epoch {epoch+1:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    # Final test evaluation
    bert.eval()
    test_correct, test_total = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)

            logits = bert(input_ids, attn_mask)
            test_correct += (logits.argmax(dim=-1) == labels).sum().item()
            test_total   += input_ids.size(0)

    test_acc = test_correct / test_total
    wandb.log({"test/acc": test_acc})
    print(f"Test acc {test_acc:.3f}")

    wandb.finish()


if __name__ == "__main__":
    train()