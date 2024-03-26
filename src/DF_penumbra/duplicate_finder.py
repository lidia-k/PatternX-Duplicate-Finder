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

    def build_duplicate_edges(self):
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
    
    def find_obvious_duplicates(self):
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