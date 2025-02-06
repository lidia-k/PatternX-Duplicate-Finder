import time

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase

from src.utils import auto_config as config

class timer:
    def __init__(self):
        self._start = time.time()
        self._end = None
        self._runtime = None

    def end(self):
        self._end = time.time()
        self._runtime = float(str(time.time() - self._start)[:5])
        return self._runtime


class Graph:
    def __init__(self, url, username, password):
        self._url = url
        self._username = username
        self._password = password

    # Helper function that runs cypher transaction on local database
    def cypher_transaction(self, cypher, parameters=None):
        driver = GraphDatabase.driver(self._url, auth=(self._username, self._password))
        values = []
        with driver.session() as session:
            res = session.run(cypher, parameters=parameters)
            for record in res:
                values.append(record.values())
        driver.close()
        return values

    # Helper function wrapped around cypher_transaction() for timing
    def query(self, cypher):
        time = timer()
        result = self.cypher_transaction(cypher)
        runtime = time.end()
        return result, runtime

    # Get type and number of each FHIR resource in the database
    def resource_metrics(self):
        cypher = f'''
            MATCH (r:resource) 
            WITH DISTINCT(r.resource_type) AS resource_types
                ORDER BY resource_types
            UNWIND resource_types as resource_type
            MATCH (r:resource)
            WHERE r.resource_type = resource_type
            WITH resource_type, COUNT(r) as resource_count
            RETURN resource_type, resource_count
                ORDER BY resource_count
        '''

        resource_count, runtime = self.query(cypher)
        return resource_count

    # Standard metrics for counting nodes and relationships
    def database_metrics(self):
        node_count = 0
        relationship_count = 0

        cypher = f'''
            MATCH (n) 
            WITH COUNT(n) as node_count
            MATCH ()-[r]->()
            WITH node_count, COUNT(r) as relationship_count
            RETURN node_count, relationship_count
        '''

        count_result, runtime = self.query(cypher)
        if (len(count_result) != 0):
            node_count = count_result[0][0]
            relationship_count = count_result[0][1]

        return node_count, relationship_count

    # Deletes all nodes and their relationships in database
    def wipe_database(self):
        node_count, relationship_count = self.database_metrics()

        cypher = f'''
            MATCH (n) DETACH DELETE n
        '''

        delete_result, runtime = self.query(cypher)
        return 'Deleted {} nodes and {} relationships in {} seconds'.format( node_count, relationship_count, runtime )
    
    def get_driver(self):
        try:
            driver = GraphDatabase.driver(self._url, auth=(self._username, self._password))
            return driver
        except Exception as e:
            print('Is the neo4j docker container running?')
            return e


class VectorGraph:

    def __init__(self, node_label, index_name, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding = HuggingFaceEmbeddings(model_name=model)
        self.node_label = node_label
        self.index_name = index_name
    
    def _retrieve_existing_index(self, query):
        return Neo4jVector.from_existing_index(
            embedding=self.embedding,
            url=config.NEO4J_URL,
            username=config.NEO4J_USER,
            password=config.NEO4J_PASSWORD,
            index_name=self.index_name,
            retrieval_query=query,
            search_type="hybrid",
            keyword_index_name="keyword"
        )

    def initialize_index(self, query=None):
        try: 
            index = self._retrieve_existing_index(query)
        except ValueError:
            Neo4jVector.from_existing_graph(
                embedding=self.embedding,
                url=config.NEO4J_URL,
                username=config.NEO4J_USER,
                password=config.NEO4J_PASSWORD,
                index_name=self.index_name,
                node_label=self.node_label,
                text_node_properties=['text'],
                embedding_node_property='embedding',
                search_type="hybrid"
            )
            index = self._retrieve_existing_index(query)
        return index 
    def similarity_search_with_score(self, text, k=4):
        return self.index.similarity_search_with_score(query=text, k=k)

