from dao.NEO4J_Graph import Graph
from utils import auto_config as config


class DuplicateFinder:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )

    def find_obvious_duplicate(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            name_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname]-(b)
                    RETURN count(*)
                    '''
            name_npi_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.npi = b.npi AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_npi]-(b)
                    RETURN count(*)
                    '''
            name_country_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.country = b.country AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_country]-(b)
                    RETURN count(*)
                    '''
            name_s_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.specialty = b.specialty AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_speciality]-(b)
                    RETURN count(*)
                    '''
            name_email_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.email = b.email AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_email]-(b)
                    RETURN count(*)
                    '''
            name_sno_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.sap_no = b.sap_no AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_sap_no]-(b)
                    RETURN count(*)
                    '''
            name_qid_q = '''
                    MATCH (a),(b)
                    WHERE a.fullname = b.fullname AND a.qb_id = b.qb_id AND id(a) < id(b)
                    MERGE (a)-[:r1_fullname_qb_id]-(b)
                    RETURN count(*)
                    '''
            dict_q = {
                'fullname': name_q,
                'fullname & npi': name_npi_q,
                'fullname & country': name_country_q,
                'fullname & specialty': name_s_q,
                'fullname & email': name_email_q,
                'fullname & sap_no': name_sno_q,
                'fullname & qb_id': name_qid_q
            }
            for k, q in dict_q.items():
                result = session.run(q)
                print(f'Found {result.single()[0]} duplicates for {k}')
    