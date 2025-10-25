import pandas as pd
import os 


def search_keyword(data, keyword):

  keyword_lower = keyword.lower()
  mask = data['body_text'].astype(str).str.lower().str.contains(keyword_lower, na=False)
  return data[mask]


country_map = {'us': "United States", 'uk': "United Kingdom"}

def print_search_results(results, keyword):

  for idx, row in results.iterrows():
    print(f"Title: {row.get('title', 'N/A')}")
    print(f"Author: {row.get('author_name', 'N/A')} ({row.get('author_born', '')}-{row.get('author_died', '')}) from {country_map.get(row.get('author_country', ''),row.get('author_country'))}")
    print(f" {str(row.get('body_text', 'N/A'))[:100]}{'...' if len(str(row.get('body_text', ''))) > 100 else ''}")
    print("-" * 50)



df = pd.read_csv('PoemData.csv')
df['author_born'] = df['author_born'].astype("Int64")
df['author_died'] = df['author_died'].astype("Int64")

test_keywords = ['sun', 'happy', 'Flower', 'COOL']

for keyword in test_keywords:
  results = search_keyword(df, keyword)
  print_search_results(results, keyword)




#if os.path.exists("PoemData.csv"):
#   df = pd.read_csv("PoemData.csv")
#   print("File loaded successfully!")
#else:
#    print("File not found. Files in current directory:")
#   print(os.listdir("."))