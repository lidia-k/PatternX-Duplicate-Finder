# Requirement:
- Python 3.9

# Duplicate Finder

The repo holds duplicate finders built for different datasets. 
- DF_adventureworks is for the experiment run on MS' adventureworks data. 
- GNN_on_FHIR is for the experiment run on the Synthea data. 
- DF_penumbra is for the duplication detection carried out on the client data. 

run: `python3 run.py --help` for more details of commands. 

## Adventure Works

### How to set up Adventure Works on Postgres, using Docker.

**Step 1.** Clone [this repo](https://github.com/lorint/AdventureWorks-for-Postgres) in `src/AdventureWorks-for-Postgres` folder

**Step 2.** Download [Adventure Works 2014](https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks-oltp-install-script.zip). It doesn't have to be the 2014 version but that’s what the above repo is using. If the database schema of other versions is different from the 2014 one, you will have to update the ruby script to convert data.

**Step 3.** Rename the zip file to `adventure_works_2014_OLTP_script.zip` to be compatible with the filename used in the dockerfile. and move this file to `src/AdventureWorks-for-Postgres` folder

**Step 4.** Run docker-compose up at the root level (`src/AdventureWorks-for-Postgres`) of the repo. It will build a postgres container with the data restored in it.

### How to run the duplicate finder

Before running the duplicate finder, you will have to process the data first. 
The current run script doesn't do it automatically, but you can simply instantiate the class as below.
The data processor currently works only for the production.product table.
```
DataProcessor(table_name='production.product')
```

Once the data is successfully processed, you can run the run script to produce vectors and do the similarity search. 
Based on what you'd like to experiment, you'll have to pick and use a relevant function of the DuplicateFinder class. 
```
python3 run.py adventureworks
```

## Synthea Data with Graph Neural Networks

### Get Synthea Data

You can download the SyntheticMass dataset [here](https://synthea.mitre.org/downloads).

### Run Neo4j 

Use the following command to run a Neo4j container. You will need a Neo4j container running to build database from the dataset and create datapoints. 

```
docker run --name testneo4j -p7474:7474 -p7687:7687 -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

### Build Neo4j database from the Synthea dataset

Run `python build_database_from_FHIR.py` from /GNN_on_FHIR directory. To run the script successfully, the neo4j container should be running locally and the following environment variables need to be set. Please double check if the dataset path is correctly set in the script. 

| Variable | Description | Value for above Docker |
|----------|-------------|------------------------|
| NEO4J_URL | Where to find the instance of Neo4j. | bolt://localhost:7687 |
| NEO4J_USER | The username for the database. | neo4j |
| NEO4J_PASSWORD | The password for the database. | password |

_TODO: The current code is creating a edge type for every single edge, which exponentially increases the total number of edge types. This part of the code (`FHIR_to_graph.py/resource_to_edges`) needs to be updated to only create a new edge type for a unique relationship between two node types._


### Create datapoints from the Neo4j database

Run `python build_datapoints_from_db` from /GNN_on_FHIR directory.
The Neo4j container that has all the data loaded should be running locally. 

Running this script successfully creates /preprocessed_datapoints directory with the pickle files in it. 
The code also creates `db_info.json` that maps all the node types and edge types to unique numbers as well as a list of features for each node type.  

## Penumbra 

### Run Neo4J

We've set up a Neo4j server on an EC2 instance, with the database already populated with data. You can access the Neo4j Browser by navigating to the server's IP address. To connect to the database, update the .env file with the appropriate configuration settings.

To run Neo4j locally, you need to pull the Neo4j Docker image and run a container. Execute the following commands in your terminal from the root directory of the project:
```
docker pull neo4j

docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/src/data:/var/lib/neo4j/import/src/data \
    -e NEO4J_AUTH=neo4j/password \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    --env NEO4J_PLUGINS='["graph-data-science", "apoc"]' \
    -d neo4j:latest
```

### Traditional ML Approach 

Before running the code, create a virtual environment and use `requirements-new.txt` to install requirements. 
(`requirements.txt` might cause some conflicts. )

**Preprocess and Load Data (Skip This Step)**

- Input: Save each sheet from the original Excel files as separate CSV files in /src/data..
- Usage: `python3 run.py --project penumbra --task neo4j`
- Method:
    1) Create an edge between two nodes if their properties exactly match. The edge types created are: npi, fullname_email, fullname_sap_no, and fullname_qb_id, identifying "obvious duplicates."
    2) For each cluster of connected nodes, create a "master node" containing all properties of the connected nodes. (The master node is created for the RAG approach.)
- Output: Check the Neo4j GUI at localhost:7474 to confirm the data is loaded correctly, showing three types of nodes (Master, Provider, Speaker) and five types of edges.

**Prepare Training Data and Train A Model**

- Input: Data stored in Neo4J.
- Usage: `python3 run.py --project penumbra --task m_training --m_model dt --npi`. 
    1) Specify which model to use by adding `--m_model <model_name>`. The model options are: Decision Tree (dt), Support Vector Machine (svm), Random Forest (rf), Logistic Regression (lg), Linear Regression (ln), and Naive Bayes (nb).
    2) To train a model with the labeled data including NPIs, add `--npi` to the command. 
- Method: 
    1) Use obvious duplicates to create matching (label 1) and non-matching (label 0) pairs of nodes. A total of 5925 pairs are labeled as 1 and 11850 pairs are labeled as 0. You can increase the number of the label 0 pairs by adjusting `skewed_factor` in the line 166 of the run file. 
    2) Train and evaluate a model using the py_entitymatching library.
- Output: Evaluation results are printed.

**Test the Model**

There are two different rounds of tests you can implement.

Test 1: 
- Input: `model.pkl`, `feature_table.pkl` and `dropped.csv` saved as a result of the training 
- Usage: `python3 run.py --project penumbra --task test1`
- Method: Use the non-matching pairs that were generated but weren’t used for the training. It’s not labeled but we know they should be predicted as 0. 
- Output: Print statement about how many of pairs are predicted as false negatives.

Test 2:
- Input: `model.pkl`, `feature_table.pkl` and Neo4J data.
- Usage: `python3 run.py --project penumbra --task test2 --npi` If the model is trained without NPIs, please remove the `--npi` flag.
- Method: Use the random data queried from Neo4J based on the following criteria:
    1) Pairs that have the same emails or sap numbers but don’t have any relationship. We used this criteria because there might be fuzzy duplicates returned from this. 
    2) Pairs that don’t have any relationship.  
- Output: `predictions.csv` including the `predicted` column. You can manually check if they are predicted right. 


### NPI Validation & RAG Approach 

**Validate NPIs and Names** 

- Input: NPI numbers and full names from Neo4J
- Usage: `python3 run.py --project penumbra --task npi`
- Method: 
    1) Verify that each NPI has 10 digits.
    2) Check the existence of the NPI and match names against the Government NPI Registry using the Levenshtein Distance from FuzzyWuzzy.
    1) Check if an NPI is 10 digits. 
    2) Check if the NPI exists and match names against [the gov NPI Registry](https://npiregistry.cms.hhs.gov/search), using Levenshtein Distance from FuzzyWuzzy.
- Output:   Creates npi_val.txt, recording all results.

**RAG with a Language Model.** 

- Usage: `DuplicateFinder().similarity_search()`
- Method: 
    1) Generate embeddings for text properties of all master nodes and other unconnected nodes using a setence transformer. 
    2) Perform similarity search by using Neo4J vector storage provided by langchain.
- Output: `similarity_search.csv`containing results of similar nodes above the score threshold of 0.96.


### Training and prediction with DeepLearning
 - training: `python run.py --project penumbra --task train`
 - prediction: `python run.py --project penumbra --task predict --model demo_model.pth --data new_data.csv`
 - online training: `python run.py --project penumbra --task online_train --model model.pth --data wrong_prediction.csv`

