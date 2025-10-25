from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
import os 
import numpy as np

def search_keywords(data, keywords):

    '''
            - Containers within a container approach 
            - Each element of a list contains the filtered list of poems for a single pass O(n) time complexity 
            - So essentially this is a dictionary...
    
    '''

    keyword_matches = {}

    for keyword in keywords:
        mask = data['body_text'].astype(str).str.lower().str.contains(keyword.lower(), na=False)
        keyword_matches[keyword] = data[mask]

    return keyword_matches

def get_embeddings(poems, query):

    '''
            - embedding the data...
    '''

    query_embeddings = model.encode(query)
    embeddings = model.encode(poems)

    return [query_embeddings, embeddings]

def format_poem_output(poem_row):
    
    title = poem_row.get('title', 'N/A')
    author = poem_row.get('author_name', 'N/A')
    author_born = poem_row.get('author_born', 'N/A')
    author_died = poem_row.get('author_died', 'N/A')
    country = poem_row.get('author_country', 'N/A')
    
    body_text = str(poem_row.get('body_text', ''))
    
    if len(body_text) > 400:
        
        formatted_text = body_text[:200] + body_text[-200:]
    else:
        
        formatted_text = body_text
    
    formatted_output = f"""Title: {title}
Author: {author}
Born: {author_born}
Died: {author_died}
Country: {country}

{formatted_text}"""
    
    return formatted_output

def get_top_matches(data, query, keywords, n):

    filtered_poems = search_keywords(data, keywords)
    
    combined_poems = pd.DataFrame()
    for keyword, poems_df in filtered_poems.items():
        combined_poems = pd.concat([combined_poems, poems_df]).drop_duplicates()
    
    poem_texts = combined_poems['body_text'].tolist()

    print("Generating embeddings...")
    query_embeddings, embeddings = get_embeddings(poem_texts, query)

    similarities = cos_sim(query_embeddings, embeddings).tolist()[0]
    
    similarities_array = np.array(similarities)
    top_indices = similarities_array.argsort()[-n:][::-1]  
    top_scores = similarities_array[top_indices]
    
    print(f"Top {len(top_indices)} similarity scores: {top_scores}")
    
    
    results = []
    for i, idx in enumerate(top_indices):
        poem_row = combined_poems.iloc[idx]
        formatted_poem = format_poem_output(poem_row)
        results.append(formatted_poem)
    
    results_string = "\n----------\n".join(results )
    print(results_string)

    return results_string




print("loading...csv")


df = pd.read_csv('PoemData.csv', encoding = "utf-8").head(20)

print("embedding language...")

# 1. Specify preferred dimensions
dimensions = 384

# 2. Load model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-xsmall-v1", truncate_dim=dimensions)

query = 'I want a poem, full of joy and flowers and sunshine'

get_top_matches(df, query, keywords=['joy', 'flowers', 'sunshine'], n=5)




'''




docs = [
    query,
]

poem_list = df['body_text'].to_list()[:100]

docs.extend(poem_list)

# 3. Encode
query_embeddings = model.encode(query)
embeddings = model.encode(docs)
similarities = cos_sim(query_embeddings, embeddings).tolist()[0]

best_match = ""
current_max = -1
for i in range(len(docs)):
    if similarities[i] > current_max and i > 0:
        current_max = similarities[i]
        best_match = f"{docs[i]} -> {embeddings[i][:5]} = {similarities[i]}"
    print(f"{docs[i]} -> {embeddings[i][:5]} = {similarities[i]}")
print(f"\n The best match was:  {best_match}")

'''