import os

import pandas as pd 
from langchain import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


raw_data = {
   'Company': ['Miscrosoft', 'Google', 'Tesla', 'Amazon'],
   'Headquarters': ['Redmond, Washington, USA', 'Mountain View, California, USA', 'Austin, Texas, USA', 'Seattle, Washington, USA'],
   'Number of employees': [238000, 178234, 127855, 1684853],
   'Revenue': ['$211.9 billion', '$279.8 billion', '$81.5 billion', '$514 billion'],
   'Key sectors': ['Technology', 'Technology and Communications', 'Consumer Cyclical, Auto Manufacturers', ' Consumer Cyclical, Internet Retail']
}
df = pd.DataFrame(raw_data)
file_path = 'company_data.csv'
df.to_csv(file_path, index=False)

loader = CSVLoader(file_path=file_path)
docs = loader.load()

# Embeddings
embeddings = HuggingFaceEmbeddings()

# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
db = FAISS.from_documents(docs, embeddings)

# QA Chain 
llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":1.0, "max_length":512})
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is the revenue of Tesla?"
results_with_score = db.similarity_search_with_score(query)
docs = db.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(f'AI answer: {answer}, Similarity score: {results_with_score[0][1]}')

query = "What are the key sectors of Google?"
results_with_score = db.similarity_search_with_score(query)
docs = db.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(f'AI answer: {answer}, Similarity score: {results_with_score[0][1]}')


query = "Which company has Technology as a key sector?"
results_with_score = db.similarity_search_with_score(query)
docs = db.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(f'AI answer: {answer}, Similarity score: {results_with_score[0][1]}')

