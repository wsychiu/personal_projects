import pandas as pd
import json
import time
from openai import OpenAI, RateLimitError

    
# Batch generator for LLM batching
def llm_batch_records(records, batch_size=20):
    for batch_num in range(0, len(records), batch_size):
        yield records.iloc[batch_num : batch_num+batch_size]

# Uses an LLM to classify a batch of articles
def llm_classify_batch(model, llm_model, system_prompt, batch):
    # Putting batch of articles into the same prompt
    article_list = ""
    for i in range(len(batch)):
        article_list += (
            f"Article: {i}.\n"
            f"Title: {batch.iloc[i]['title']}\n"
            f"Text: {batch.iloc[i]['text']}\n\n"
        )
    
    # Classifying
    classify_response = model.responses.create(
        model = llm_model,
        input = [
            {
                'role' : 'system',
                'content' : (system_prompt)
            },
            {
                'role' : 'user',
                'content' : (article_list)
            }
        ],
        temperature=0
    )

    return classify_response

# Iterates over the batches of articles and classifies them
def llm_classify(batches, llm_model='gpt-5.4-mini', max_tries=5):
    token_usage = {
        'input' : 0,
        'output' : 0
    }
    llm_client = OpenAI()
    system_prompt = '''Your task is to produce an analysis of a news article and provide reasons why the news article is likely Real or Fake news.
        You will be provided the Title and Text of a list of news article and you must provide your response in the following JSON list format:
        [{"Article" : int, "Classification" : "REAL" or "FAKE"}]  
        '''
    
    llm_classifications = pd.DataFrame()

    for i, batch in enumerate(batches):
        delay = 2 # seconds of delay before retrying in case of rate limit error

        batch_size = batch.shape[0]
        print(f'Classifying Batch: {i+1} | {batch_size} articles')
        
        for attempt in range(max_tries):
            try:
                classify_response = llm_classify_batch(
                    llm_client,
                    llm_model,
                    system_prompt,
                    batch
                )
                break
            except RateLimitError:
                print(f' >> API Token Rate Limit hit on attempt # {attempt + 1}. Sleeping for {delay}s')
                time.sleep(delay)
                delay *=2 # Exponential back-off
        
        # Reading response in JSON - easiest to do in DataFrame
        y_pred = (
            pd.DataFrame(json.loads(classify_response.output_text))
             ['Classification']
            .map({'REAL':0, 'FAKE':1})
            .rename('llm_y_pred')
        )

        # Preserving index to support matching back with target
        y_pred.index = batch.index
        llm_classifications = pd.concat([llm_classifications, y_pred], axis=0)

        token_usage['input'] += classify_response.usage.input_tokens
        token_usage['output'] += classify_response.usage.output_tokens

        #raise RuntimeError(f"Rate Limit Error hit - even after {max_tries} attmempts")

    return llm_classifications, token_usage