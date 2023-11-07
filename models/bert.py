import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

class EmotionClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-emotion')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased-emotion')
        self.optimizer = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def finetune(self, X_train, y_train, batch_size=16, learning_rate=2e-5, num_epochs=3):
        # Tokenize the input data and create DataLoader objects
        train_encodings = self.tokenizer(X_train, truncation=True, padding=True, return_tensors='pt', max_length=128)
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Set up optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

    def predict(self, X_test):
        # Tokenize the input test data and create DataLoader objects
        test_encodings = self.tokenizer(X_test, truncation=True, padding=True, return_tensors='pt', max_length=128)
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Evaluation
        self.model.eval()
        y_pred = []

        for batch in test_loader:
            input_ids, attention_mask = batch
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        return y_pred

    def evaluate_acc(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)
