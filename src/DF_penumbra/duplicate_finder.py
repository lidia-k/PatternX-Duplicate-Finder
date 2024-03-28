import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector

from dao.NEO4J_Graph import Graph
from utils import auto_config as config

EDGE_TYPES = [
    'fullname',
    'npi',
    'email',
    'fullname_npi',
    'fullname_email',
    'fullname_sap_no',
    'fullname_qb_id'
]

class DuplicateFinder:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        for node in ['Provider', 'Speaker']:
            self.vector_graph = Neo4jVector.from_existing_graph(
                embedding=self.embeddings,
                url=config.NEO4J_URL,
                username=config.NEO4J_USER,
                password=config.NEO4J_PASSWORD,
                index_name='penumbra_index',
                node_label=node,
                text_node_properties=['text'],
                embedding_node_property='embedding'
            )

    def build_duplicate_edges(self):
        """
        If two nodes have the same property, create an edge between them.
        """
        driver = self.graph.get_driver()
        with driver.session() as session:
            for type in EDGE_TYPES:
                if '_' not in type:
                    base_q = f'''
                            MATCH (a),(b)
                            WHERE a.{type} = b.{type} AND id(a) < id(b)
                            MERGE (a)-[:r1_{type}]-(b)
                            RETURN count(*)
                            '''
                else: 
                    p1, p2 = type.split('_', 1)
                    base_q = f'''
                            MATCH (a),(b)
                            WHERE a.{p1} = b.{p1} AND a.{p2} = b.{p2} AND id(a) < id(b)
                            MERGE (a)-[:r1_{type}]-(b)
                            RETURN count(*)
                            '''
                result = session.run(base_q)
                print(f'Found {result.single()[0]} duplicates for {type}')
    
    def lookup_obvious_duplicates(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            for type in EDGE_TYPES:
                q = f'''
                    MATCH (a)-[:r1_{type}]-(b)
                    WHERE id(a) < id(b)
                    RETURN a, b
                    '''
                result = session.run(q).data()
                print(f'Found {len(result)} duplicates for {type}')
    
    def similarity_search(self):
        csv_data = []
        cols = ['uid', 'fname', 'lname', 'fullname', 'npi', 
                'country', 'speciality', 'email', 'sap_no', 'qb_id']
        empty_row = {col: '' for col in cols}

        driver = self.graph.get_driver()
        with driver.session() as session:
            for node_type in ['Provider', 'Speaker']:
                q = f'MATCH (n:{node_type}) RETURN n'
                result = session.run(q).data()
                print(f'Found {len(result)} nodes for {node_type}')
                
                for node in result:
                    node = node['n']
                    q = node['text']
                    uid = node['uid']

                    csv_data.append(empty_row)    

                    node_dict = {}
                    for col in cols:
                        node_dict['score'] = ''
                        node_dict[col] = node.get(col)
                    csv_data.append(node_dict)

                    results = self.vector_graph.similarity_search_with_score(q)
                    for result in results:
                        doc, score = result
                        metadata = doc.metadata
                        doc_id = metadata.get('uid')

                        if score > 0.96 and uid != doc_id:
                            doc_dict = {}
                            for col in cols:
                                doc_dict['score'] = score
                                doc_dict[col] = metadata.get(col)
                            csv_data.append(doc_dict)

            df = pd.DataFrame(csv_data)
            df.fillna('', inplace=True)
            df.to_csv('similarity_search.csv', index=False)
                                