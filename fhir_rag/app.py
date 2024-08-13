import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOllama
from langchain.llms import Ollama

from src.dao.NEO4J_Graph import VectorGraph


"""
Question examples:
- How much did the cesarean section on Jan. 16, 2014 cost?
- From which institution, does the patient receive the cesarean section on Jan. 16, 2014?
- When did the patient get the cesarean section? 
"""

app = Flask(__name__)


# Initialize VectorGraph and index
vg = VectorGraph(node_label='resource', index_name='fhir_index')
contextualized_query = """
match (node)<-[]->(sc:resource)
with node.text as self, reduce(s="", item in collect(distinct sc.text) | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata limit 1
return "Primary Entry:\n" + self + ctxt as text, score, metadata
"""
index = vg.initialize_index(contextualized_query)

prompt = '''
System: The context below contains entries about the patient's healthcare. 
Please limit your answer to the information provided in the context. Do not make up facts. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If you are asked about the patient's name and one the entries is of type patient, you should look for the first given name and family name and answer with: [given] [family]
----------------
{context}
Human: {question}
'''
prompt = PromptTemplate.from_template(prompt)

ollama_model = 'mistral'
llm = Ollama(model=ollama_model)

k_nearest = 200
vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOllama(model=ollama_model),
    chain_type="stuff",
    retriever=index.as_retriever(search_kwargs={'k': k_nearest}),
    verbose=True,
    chain_type_kwargs={"verbose": True, "prompt": prompt}
)

@app.route('/', methods=['GET', 'POST'])
def index():
    no_context_answer = None
    context_answer = None
    if request.method == 'POST':
        question = request.form['question']
        no_context_answer = llm(question)
        context_answer = vector_qa.run(question)
    return render_template('index.html', no_context_answer=no_context_answer, context_answer=context_answer)

if __name__ == '__main__':
    app.run(debug=True)