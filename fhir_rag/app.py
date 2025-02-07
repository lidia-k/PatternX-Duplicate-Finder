import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub, Ollama
from langchain.chat_models import ChatOllama

from src.dao.NEO4J_Graph import VectorGraph


type_model = os.getenv("TYPE_MODEL") or "huggingface"
app = Flask(__name__)

if type_model == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
else:
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not huggingface_api_key:
        raise ValueError("Please set the HUGGINGFACE_API_KEY environment variable")


# Initialize VectorGraph and index
# note: need to delete index when you want to change the model or update data in neo4j: DROP INDEX fhir_index
vg = VectorGraph(
    node_label="resource", index_name="fhir_index", model="NovaSearch/stella_en_1.5B_v5"
)  # model='NovaSearch/stella_en_1.5B_v5' / BAAI/bge-large-en-v1.5
contextualized_query = """
match (node)<-[]->(sc:resource)
with node.text as self, reduce(s="", item in collect(distinct sc.text)[..5] | s + "\n\nSecondary Entry:\n" + item ) as ctxt, score, {} as metadata 
return "Primary Entry:\n" + self + ctxt as text, score, metadata
"""
index = vg.initialize_index(contextualized_query)


prompt = """
System: The following information contains entries about the patient.
Use the primary entry and then the secondary entries to answer the user's question.
Each entry is its own type of data and secondary entries are supporting data for the primary one.
Please limit your answer to the information provided in the context. Do not make up facts.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If you are asked about the patient's name and one the entries is of type patient, you should look for the first given name and family name and answer with: [given] [family]
----------------
{context}
Human: {question}
"""

prompt = PromptTemplate.from_template(prompt)

# ollama_model = 'mistral'
# llm = Ollama(model=ollama_model)
# chat_model = ChatOllama(model=ollama_model)

if type_model == "openai":
    chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
else:
    chat_model = HuggingFaceHub(
        repo_id="deepseek-ai/DeepSeek-R1",
        huggingfacehub_api_token=huggingface_api_key,
        model_kwargs={"temperature": 0.6},
    )

k_nearest = 10
vector_qa = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=index.as_retriever(search_kwargs={"k": k_nearest}),
    verbose=True,
    chain_type_kwargs={"verbose": True, "prompt": prompt},
)


"""
Download file: https://github.com/synthetichealth/synthea-sample-data/blob/1fe1beaa80a8fbe7b64c0c135bcbb8b1346ef38a/downloads/latest/synthea_sample_data_fhir_latest.zip
and import file: Alfonso758_Bins636_e80d4c62-149a-a6a6-4b39-9d4aa3e07ba7.json
Question examples:
- What can you tell me about the medical claim created on March 06, 1977?
- Based on this explanation of benefits, how much did it cost and what service was provided?
- How much did the colon scan on Jan. 18, 2014 cost?
"""


@app.route("/", methods=["GET", "POST"])
def index():
    no_context_answer = None
    context_answer = None
    if request.method == "POST":
        question = request.form["question"]
        # no_context_answer = llm(question)
        with get_openai_callback() as cb:
            context_answer = vector_qa.run(question)
            if type_model == "huggingface":
                # Hack with HuggingFaceHub: output always includes prompt and answer
                # get data after "AI:" or "Assistant:" or "Answer:"
                arr_answer = ["AI:", "Assistant:", "Answer:"]
                for answer in arr_answer:
                    if answer in context_answer:
                        context_answer = context_answer.split(answer)[1].strip()
                        break

            token_usage = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost,
            }
            print(token_usage)
    return render_template(
        "index.html", no_context_answer=no_context_answer, context_answer=context_answer
    )


if __name__ == "__main__":
    app.run(debug=os.getenv("DEBUG") or True)
