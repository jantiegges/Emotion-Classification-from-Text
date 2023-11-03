import itertools
import pandas as pd

from utils.data_preparation import naive_bayes_preprocessing
from models.naive_bayes import NaiveBayes

def nb_grid_search(params):
    results_list = []
    param_combinations = itertools.product(params['alpha'], params['fit_prior'], params['use_bigrams'], params['remove_stopwords'])

    for alpha, fit_prior, use_bigrams, remove_stopwords in param_combinations:
        processed_data, _ = naive_bayes_preprocessing(
            remove_stopwords=remove_stopwords, 
            use_bigrams=use_bigrams
        )
        X_train, y_train = processed_data['train']
        X_val, y_val = processed_data['validation']
        
        nb = NaiveBayes(alpha=alpha, fit_prior=fit_prior)
        nb.fit(X_train, y_train)
        predictions = nb.predict(X_val)
        acc = nb.evaluate_acc(y_val, predictions)
        
        results_list.append({
            'alpha': alpha,
            'fit_prior': fit_prior,
            'use_bigrams': use_bigrams,
            'remove_stopwords': remove_stopwords,
            'val_accuracy': acc
        })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='val_accuracy', ascending=False).reset_index(drop=True)
    results_df.to_pickle('out/nb_grid_search_results.pkl')
    return results_df

if __name__ == '__main__':
    params = {
        'alpha': [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        'fit_prior': [True, False],
        'use_bigrams': [True, False],
        'remove_stopwords': [True, False]
    }
    num_combinations = len(list(itertools.product(*params.values())))
    print(f"The number of combinations is: {num_combinations}")
    grid_search_results = nb_grid_search(params)