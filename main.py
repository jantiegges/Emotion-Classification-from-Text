import itertools
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from utils.data_preparation import naive_bayes_preprocessing, bert_preprocessing
from models.naive_bayes import NaiveBayes
from models.bert import Bert


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
        
        nb = NaiveBayes()
        nb.fit(X_train, y_train, alpha_prior=alpha_prior, alpha_likelihood=alpha_likelihood)
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

        bert = Bert()
        bert.fit(train_loader, epochs, lr, weight_decay=weight_decay, fine_tune_last_layers=fine_tune_last_layers)
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


if __name__ == '__main__':

    nb_search = False
    bert_search = False

    # Naive Bayes Grid Search
    if nb_search:
        params = {
            'alpha_prior': [0, 1, 10, 100, 1000, 10000, 100000],
            'alpha_likelihood': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
            'use_bigrams': [True, False],
            'remove_stopwords': [True, False]
        }
        num_combinations = len(list(itertools.product(*params.values())))
        print(f"The number of combinations is: {num_combinations}")
        grid_search_results = nb_grid_search(params)

    # BERT grid search
    if bert_search:
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
