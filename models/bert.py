import torch
from transformers import BertModel, AdamW
# from transformers import pipeline
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

class Bert:
    def __init__(self):
        self.model = BertModel.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
        self.optimizer = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def fit(self, input_ids, attention_mask, y_train, batch_size=16, learning_rate=2e-5, num_epochs=3):
        # Create DataLoader objects directly from pre-tokenized data
        train_dataset = TensorDataset(input_ids, attention_mask, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                batch_input_ids, batch_attention_mask, batch_labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

    def predict(self, input_ids, attention_mask):
        # Create DataLoader objects directly from pre-tokenized data
        test_dataset = TensorDataset(input_ids, attention_mask)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Evaluation
        self.model.eval()
        y_pred = []

        for batch in test_loader:
            batch_input_ids, batch_attention_mask = batch
            with torch.no_grad():
                outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs.logits
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        return y_pred

    def evaluate_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
if __name__ == "__main__":
    # Initialize the Bert model
    bert_model = Bert()
