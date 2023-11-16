import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm


class Bert:
    """Class for implementing the three BERT models"""

    def __init__(self, model_version='bhadresh-savani/bert-base-uncased-emotion'):

        self.model = BertForSequenceClassification.from_pretrained(model_version, output_attentions=True)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def fit(self, train_loader, epochs, lr, weight_decay=0.01, fine_tune_last_layers=False):

        if fine_tune_last_layers:
            for param in self.model.bert.parameters():
                param.requires_grad = False

        # set up optimizer and scheduler
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.model.device) for t in batch)
                self.model.zero_grad()

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()

                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"  Average training loss: {avg_train_loss}")

        return self

    def predict(self, data_loader, return_attentions=False):

        self.model.eval()
        y_pred = []
        all_attentions = [] if return_attentions else None

        with torch.no_grad():
            for batch in data_loader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(self.model.device) for t in batch)
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                logits = outputs.logits
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

                if return_attentions:
                    attentions = outputs.attentions
                    all_attentions.extend(attentions)

        return (y_pred, all_attentions) if return_attentions else y_pred

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)
