# Dulicate Finder

The repo holds duplicate finders built for different datasets. 
- DF_adventureworks is for the experiment run on MS' adventureworks data. 
- GNN_on_FHIR is for the experiment run on the Synthea data. 
- DF_penumbra is for the duplication detection carried out on the client data. 

## Adventure Works

### How to set up Adventure Works on Postgres, using Docker.

**Step 1.** Clone [this repo](https://github.com/lorint/AdventureWorks-for-Postgres) 

**Step 2.** Download [Adventure Works 2014](https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorks-oltp-install-script.zip). It doesn't have to be the 2014 version but that’s what the above repo is using. If the database schema of other versions is different from the 2014 one, you will have to update the ruby script to convert data.

**Step 3.** Rename the zip file to `adventure_works_2014_OLTP_script.zip` to be compatible with the filename used in the dockerfile.

**Step 4.** Run docker-compose up at the root level of the repo. It will build a postgres container with the data restored in it.

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

### Load the data in csv files to Neo4j

You first need to pull a Neo4j docker image and run a docker container for Neo4j.
```
docker pull neo4j

docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/var/lib/neo4j/import/data \
    -e NEO4J_AUTH=neo4j/password \
    -d neo4j:latest
```

Then, you can run the run script with the argument. 
When successfully run, you should see the names of the original and updated files printed on your terminal.
You can check Neo4j GUI (localhost:7474) to see there is data loaded properly.  
```
python3 run.py penumbra
```

### Validate NPIs against the government registry

### Find duplicates 

1. Obvious duplicates
2. Use a language model to generate vectors for each row. The obvious duplicates are removed (fullname + npi, fullname + email)