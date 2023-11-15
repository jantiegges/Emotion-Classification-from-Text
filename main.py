import itertools
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import pickle 
from sklearn.metrics import accuracy_score

from utils.data_preparation import naive_bayes_preprocessing, bert_preprocessing
from models.naive_bayes import NaiveBayes
from models.bert import Bert

SEED = 42
torch.manual_seed(SEED)

def nb_grid_search(params):
    results_list = []
    param_combinations = itertools.product(params['alpha_prior'], params['alpha_likelihood'], params['use_bigrams'], params['remove_stopwords'])

    for alpha_prior, alpha_likelihood, use_bigrams, remove_stopwords in param_combinations:
        processed_data, _ = naive_bayes_preprocessing(
            remove_stopwords=remove_stopwords, 
            use_bigrams=use_bigrams
        )
        X_train, y_train = processed_data['train']
        X_val, y_val = processed_data['validation']
        
        nb = NaiveBayes(alpha_prior=alpha_prior, alpha_likelihood=alpha_likelihood)
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_val)
        acc = nb.evaluate_acc(y_val, predictions)
        
        results_list.append({
            'alpha_prior': alpha_prior,
            'alpha_likelihood': alpha_likelihood,
            'use_bigrams': use_bigrams,
            'remove_stopwords': remove_stopwords,
            'val_accuracy': acc
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    results_df.to_pickle('out/nb_grid_search_results.pkl')
    return results_df

def bert_grid_search(params, exp_name):

    ## Data Preparation ##
    bert_processed_data, tokenizer = bert_preprocessing()
    input_ids_train, attention_mask_train, y_train = bert_processed_data['train']
    input_ids_val, attention_mask_val, y_val = bert_processed_data['validation']
    input_ids_test, attention_mask_test, y_test = bert_processed_data['test']

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)
    test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test)

    results_list = []
    param_combinations = itertools.product(params['epochs'], params['batch_size'], params['lr'], params['weight_decay'],
                                           params['fine_tune_last_layers'])

    for epochs, batch_size, lr, weight_decay, fine_tune_last_layers in param_combinations:

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader_eval = DataLoader(train_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        bert = Bert(fine_tune_last_layers=fine_tune_last_layers)
        bert.fit(train_loader, train_loader_eval, val_loader, y_train, y_val, epochs, lr, weight_decay)
        y_pred = bert.predict(val_loader)
        val_accuracy = accuracy_score(y_val, y_pred)

        results_list.append({
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'fine_tune_last_layers': fine_tune_last_layers,
            'val_accuracy': val_accuracy
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    results_df.to_pickle(f'out/_grid_search_results_{exp_name}.pkl')
    return results_df

def bert_train(train_loader, train_loader_eval, val_loader, epochs, lr, fine_tune_last_layers=False):
    """Fine-tune BERT for emotion classification"""

    # Load model
    model_version = 'bhadresh-savani/bert-base-uncased-emotion'
    model = BertForSequenceClassification.from_pretrained(model_version, output_attentions=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # freeze bert layers if fine-tuning only the last classifier layers
    if fine_tune_last_layers:
        for param in model.bert.parameters():
            param.requires_grad = False

    # set up optimizer and scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(model.device) for t in batch)
            model.zero_grad()        

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        # evaluate on validation set
        model.eval()
        total_eval_loss = 0
        val_pred = []

        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}"):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(model.device) for t in batch)
            
            with torch.no_grad():        
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            loss = outputs.loss
            total_eval_loss += loss.item()
            logits = outputs.logits
            val_pred.extend(logits.argmax(dim=1).cpu().numpy())

        avg_val_loss = total_eval_loss / len(val_loader)

        train_pred = bert_predict(model, train_loader_eval, return_attentions=False)

        acc_train = accuracy_score(y_train, train_pred)
        acc_val = accuracy_score(y_val, val_pred)

        print(f"  Average training loss: {avg_train_loss}")
        print(f"  Validation Loss: {avg_val_loss}")
        print(f"  Training Accuracy: {acc_train}")
        print(f"  Validation Accuracy: {acc_val}")

    # save the model to a file
    if fine_tune_last_layers:
        model_save_path = f"out/bert_last_layers.bin"
    else:
        model_save_path = f"out/bert_entire_model.bin"

    torch.save(model.state_dict(), model_save_path)

    return model

def bert_predict(model, data_loader, return_attentions=False):
    model.eval()
    y_pred = []
    all_attentions = [] if return_attentions else None

    with torch.no_grad():
        for batch in data_loader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(model.device) for t in batch)
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

            if return_attentions:
                attentions = outputs.attentions
                all_attentions.extend(attentions)

    return (y_pred, all_attentions) if return_attentions else y_pred


if __name__ == '__main__':

    ### Naive Bayes Grid Search ###
    #params = {
    #    'alpha_prior': [0, 1, 10, 100, 1000, 10000, 100000],
    #    'alpha_likelihood': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
    #    'use_bigrams': [True, False],
    #    'remove_stopwords': [True, False]
    #}
    #num_combinations = len(list(itertools.product(*params.values())))
    #print(f"The number of combinations is: {num_combinations}")
    #grid_search_results = nb_grid_search(params)

    ### BERT Grid search - last layers only ###
    params = {
        'epochs': [3],
        'batch_size': [32, 64, 128],
        'lr': [1e-5, 2e-5, 3e-5],
        'weight_decay': [0.01, 0.001],
        'fine_tune_last_layers': [True]
    }
    num_combinations = len(list(itertools.product(*params.values())))
    print(f"The number of combinations is: {num_combinations}")
    grid_search_results = bert_grid_search(params, exp_name='last_layers')
    exit()

    ### BERT ###

    ## Data Preparation ##
    bert_processed_data, tokenizer = bert_preprocessing()
    input_ids_train, attention_mask_train, y_train = bert_processed_data['train']
    input_ids_val, attention_mask_val, y_val = bert_processed_data['validation']
    input_ids_test, attention_mask_test, y_test = bert_processed_data['test']

    train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train)
    val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val)
    test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_eval = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ## Out-of-the-box BERT ##
    model_version = 'bhadresh-savani/bert-base-uncased-emotion'
    model = BertForSequenceClassification.from_pretrained(model_version, output_attentions=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    predictions, attentions = bert_predict(model, test_loader, return_attentions=True)

    with open('out/bert_oob_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open('out/bert_oob_attentions.pkl', 'wb') as f:
        pickle.dump(attentions, f)


    ## Fine-tuned BERT (entire model) ##
    epochs = 3
    lr = 2e-5

    model = bert_train(train_loader, train_loader_eval, val_loader, epochs, lr, fine_tune_last_layers=False)

    predictions, attentions = bert_predict(model, test_loader, return_attentions=True)

    with open('out/bert_entire_model_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open('out/bert_entire_model_attentions.pkl', 'wb') as f:
        pickle.dump(attentions, f)


    ## Fine-tuned BERT (last layers) ##
    epochs = 3
    lr = 2e-5

    model = bert_train(train_loader, train_loader_eval, val_loader, epochs, lr, fine_tune_last_layers=True)

    predictions, attentions = bert_predict(model, test_loader, return_attentions=True)

    with open('out/bert_last_layers_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    with open('out/bert_last_layers_attentions.pkl', 'wb') as f:
        pickle.dump(attentions, f)
