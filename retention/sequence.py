"""Matched LSTM/GRU encoders with learned count embedding and packed padding."""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler


def collate(batch):
    seq, y = zip(*batch)
    tensors = [torch.as_tensor(s, dtype=torch.float32) for s in seq]
    return pad_sequence(tensors, batch_first=True), torch.tensor([len(s) for s in seq]), torch.tensor(y, dtype=torch.float32)


class RecurrentEncoder(nn.Module):
    def __init__(self, hidden=24, kind="lstm"):
        super().__init__()
        self.embedding = nn.Linear(4, 12)
        if kind not in ("lstm", "gru"):
            raise ValueError(f"Unsupported recurrent architecture: {kind}")
        self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(16, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        x = torch.cat([torch.relu(self.embedding(x[:, :, :4])), x[:, :, 4:]], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(hidden[-1]).squeeze(-1)


# Previous local smoke bundles referenced this class name.
LSTM = RecurrentEncoder


class SequenceModel:
    def __init__(self, seed=42, hidden=24, batch_size=128, learning_rate=.002, kind="lstm"):
        self.seed, self.hidden, self.batch_size, self.learning_rate = seed, hidden, batch_size, learning_rate
        self.kind = kind

    def fit(self, sequences, y, validation=None, epochs=12, patience=3):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        self.scaler = StandardScaler().fit(np.concatenate(sequences))
        transformed = [self.scaler.transform(s).astype(np.float32) for s in sequences]
        self.model = RecurrentEncoder(self.hidden, self.kind)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        pos = float(np.sum(y)); neg = len(y) - pos
        if min(pos, neg) <= 0:
            raise ValueError("Recurrent model requires both classes")
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos))
        loader = DataLoader(list(zip(transformed, y)), batch_size=self.batch_size, shuffle=True, collate_fn=collate,
                            generator=torch.Generator().manual_seed(self.seed))
        self.history, best, wait, state = [], float("inf"), 0, None
        self.best_epoch = epochs
        for epoch in range(1, epochs + 1):
            self.model.train(); losses = []
            for x, lengths, target in loader:
                optimizer.zero_grad(); loss = loss_fn(self.model(x, lengths), target)
                loss.backward(); nn.utils.clip_grad_norm_(self.model.parameters(), 5.); optimizer.step()
                losses.append(float(loss.detach()))
            item = {"epoch": epoch, "train_weighted_loss": float(np.mean(losses))}
            if validation is not None:
                val_seq, val_y = validation
                p = self.predict_proba(val_seq)
                # Early stopping uses unweighted probability log loss, never the outer fold/test.
                val_loss = float(-np.mean(val_y * np.log(p) + (1 - val_y) * np.log(1 - p)))
                item["validation_log_loss"] = val_loss
                if val_loss < best - 1e-5:
                    best, wait, state, self.best_epoch = val_loss, 0, copy.deepcopy(self.model.state_dict()), epoch
                else:
                    wait += 1
            self.history.append(item)
            if validation is not None and wait >= patience:
                break
        if state is not None:
            self.model.load_state_dict(state)
        return self

    def predict_proba(self, sequences):
        self.model.eval()
        seq = [self.scaler.transform(s).astype(np.float32) for s in sequences]
        loader = DataLoader(list(zip(seq, np.zeros(len(seq)))), batch_size=self.batch_size, collate_fn=collate)
        result = []
        with torch.no_grad():
            for x, lengths, _ in loader:
                result.extend(torch.sigmoid(self.model(x, lengths)).numpy().tolist())
        return np.clip(np.asarray(result), 1e-7, 1 - 1e-7)

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "hidden": self.hidden, "kind": self.kind, "mean": self.scaler.mean_.tolist(),
                    "scale": self.scaler.scale_.tolist(), "seed": self.seed, "history": self.history,
                    "best_epoch": self.best_epoch}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        result = cls(seed=checkpoint["seed"], hidden=checkpoint["hidden"], kind=checkpoint["kind"])
        result.model = RecurrentEncoder(result.hidden, result.kind)
        result.model.load_state_dict(checkpoint["state_dict"])
        result.scaler = StandardScaler()
        result.scaler.mean_ = np.asarray(checkpoint["mean"], dtype=float)
        result.scaler.scale_ = np.asarray(checkpoint["scale"], dtype=float)
        result.scaler.var_ = result.scaler.scale_ ** 2
        result.scaler.n_features_in_ = len(result.scaler.mean_)
        result.history, result.best_epoch = checkpoint["history"], checkpoint["best_epoch"]
        return result
