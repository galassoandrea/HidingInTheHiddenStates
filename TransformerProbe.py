import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from torch.utils.data import Dataset
from sklearn.metrics import roc_curve, auc
import numpy as np


class EmbeddingDataset(Dataset):
    """Dataset for embeddings and labels"""

    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings).unsqueeze(1) # Add sequence dimension
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

class TransformerProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        cfg = HookedTransformerConfig(
            n_layers=1,
            d_model=input_dim,  # Embeddings dimension
            d_head=64,
            n_heads=2,
            d_mlp=input_dim * 2,
            n_ctx=1,  # Context length (we only process one embedding at a time)
            act_fn="gelu",
            normalization_type="LN",
            d_vocab=2,  # Binary classification (not used since we input embeddings)
            d_vocab_out=2,  # Binary output
            attention_dir="causal",
            use_attn_result=True,
            use_split_qkv_input=True,
            use_hook_tokens=False
        )
        self.model = HookedTransformer(cfg)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, train_loader, val_loader, epochs=5):

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):

            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_embeddings, batch_labels in train_loader:
                x = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()

                for block in self.model.blocks:
                    x = block(x)
                x = self.model.ln_final(x)
                logits = self.model.unembed(x.squeeze(1))
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == batch_labels).sum().item()
                train_total += batch_labels.size(0)

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            # Validation
            val_loss, val_acc = self.evaluate(val_loader)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}", end="\n")

        # Load best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")

    def evaluate(self, test_loader, return_predictions=True):
        """
        Evaluate the model on test data
        """
        self.model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch_embeddings, batch_labels in test_loader:
                x = batch_embeddings.to(self.device)
                batch_labels = batch_labels.to(self.device)

                for block in self.model.blocks:
                    x = block(x)
                x = self.model.ln_final(x)
                logits = self.model.unembed(x.squeeze(1))
                loss = self.criterion(logits, batch_labels)
                test_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                test_correct += (predicted == batch_labels).sum().item()
                test_total += batch_labels.size(0)

            test_loss /= len(test_loader)
            test_acc = test_correct / test_total

        return test_loss, test_acc

    def predict(self, test_loader):
        """
        Get predictions for input data
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for batch_embeddings, _ in test_loader:
                x = batch_embeddings.to(self.device)

                for block in self.model.blocks:
                    x = block(x)
                x = self.model.ln_final(x)
                logits = self.model.unembed(x.squeeze(1))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability of the positive class
                all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)