from dao.NEO4J_Graph import Graph
from utils import auto_config as config


class DuplicateFinder:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )

    def find_duplicates_by_fullname(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            query = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname]->(b)
                    RETURN count(*)
                    '''
            result = session.run(query)
            print(f'Found {result.single()[0]} duplicates by fullname')
    
    def find_duplicates_by_fullname_and_npi(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            query = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.npi = b.npi AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_npi]->(b)
                    RETURN count(*)
                    '''
            result = session.run(query)
            print(f'Found {result.single()[0]} duplicates by fullname and NPI')