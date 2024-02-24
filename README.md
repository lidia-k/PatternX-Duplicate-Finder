# Duplicate Finder on Synthea Data with Graph Neural Networks

The code to run a duplicate finder for the Synthea Data is in /GNN_on_FHIR directory.

## Get Synthea Data

You can download the SyntheticMass dataset [here](https://synthea.mitre.org/downloads).

## Run Neo4j 

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

## Build Neo4j database from the Synthea dataset

Run `python build_database_from_FHIR.py` from /GNN_on_FHIR directory. To run the script successfully, the neo4j container should be running locally and the following environment variables need to be set. Please double check if the dataset path is correctly set in the script. 

| Variable | Description | Value for above Docker |
|----------|-------------|------------------------|
| NEO4J_URL | Where to find the instance of Neo4j. | bolt://localhost:7687 |
| NEO4J_USER | The username for the database. | neo4j |
| NEO4J_PASSWORD | The password for the database. | password |

_TODO: The current code is creating a edge type for every single edge, which exponentially increases the total number of edge types. This part of the code (`FHIR_to_graph.py/resource_to_edges`) needs to be updated to only create a new edge type for a unique relationship between two node types._

## Create datapoints from the Neo4j database

Run `python build_datapoints_from_db` from /GNN_on_FHIR directory.
The Neo4j container that has all the data loaded should be running locally. 

Running this script successfully creates /preprocessed_datapoints directory with the pickle files in it. 
The code also creates `db_info.json` that maps all the node types and edge types to unique numbers as well as a list of features for each node type.  


