import argparse
import pickle
import pandas as pd

from utils import (
    TextProcessing,
    get_timestamp,
    load_data
)
from llm_inference import (
    llm_batch_records,
    llm_classify
)

class LogRegClassifier:
    def __init__(self, fpath):
        self.fpath = fpath
        self.model = pickle.load(open(fpath, "rb"))
    
    def predict(self, data:  pd.DataFrame):
        y_pred = self.model.predict(data)

        return y_pred

def lr_classify(df, fpath_model):
    # Preprocess data
    text_processor = TextProcessing(lemma=True, batch_size=500)
    lr_classifier = LogRegClassifier(fpath_model)
    
    df_processed = text_processor.transform(df)
    lr_y_pred = lr_classifier.predict(df_processed)

    df_processed['Predicted CLassification'] = lr_y_pred

    return df_processed

def logistic_regression(data, model, dirpath_output):
    df_lr_classified = lr_classify(data, model)
    lr_output_fname = f'lr_{get_timestamp()}.csv'
    df_lr_classified.to_csv(dirpath_output + lr_output_fname)

    return None

def gen_ai(data, llm_model, dirpath_output):
    text_processor = TextProcessing(lemma=False, batch_size=500)
    processed_data = text_processor.transform(data)

    batches = llm_batch_records(processed_data, batch_size=50)
    df_llm_classified, _ = llm_classify(batches, llm_model)
    llm_output_fname = f'llm_{get_timestamp()}.csv'
    df_llm_classified.to_csv(dirpath_output + llm_output_fname)

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--run_llm', required=False, action='store_true')
    args = parser.parse_args()

    fpath_model = 'models/lr_model.pkl'
    dirpath_data = 'datasets/'
    dirpath_output = 'results/'

    data = args.data
    run_llm = args.run_llm
    
    df_data = load_data(dirpath_data + data)
    print(f'Logistic Regression...')
    logistic_regression(
        df_data, 
        fpath_model,
        dirpath_output
    )
    print(f'...Logistic Regression Completed')

    if run_llm: 
        print(f'Gen AI...')
        gen_ai(df_data, 'gpt-5.4-mini', dirpath_output)
        print(f'...Gen AI Completed')