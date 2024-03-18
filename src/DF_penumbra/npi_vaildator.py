import requests
from fuzzywuzzy import fuzz

from dao.NEO4J_Graph import Graph
from utils import auto_config as config


class NPIValidator:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD
        )
        self.api_url = "https://npiregistry.cms.hhs.gov/api/"

    def check_against_gov_registry(self, npi):
        params = {"number": npi, "version": 2.1}
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json().get("results", [])
        except Exception as e:
            print(f"Error checking NPI {npi}: {e}")
            return None
        
    def _check_digits(self, session):
        # Check for NPIs that are not 10 digits
        query = '''
            MATCH (n) 
            WHERE n.npi IS NOT NULL AND size(toString(n.npi)) <> 10 
            RETURN n.npi, n.id
            '''
        result = session.run(query).data()
        print(f'NPIs larger than 10 digits: {result}')   

    def validate_NPIs(self, results):
        driver = self.graph.get_driver()
        with driver.session() as session:
            self._check_digits(session)

            query = '''
                    MATCH (n) 
                    WHERE n.npi IS NOT NULL AND size(toString(n.npi)) = 10 
                    RETURN n.npi, n.id, n.fname, n.lname, n.fullname
                    '''
            results = session.run(query).data()
            with open('npi_val.txt', 'w') as output_f:
                for result in results:
                    npi, id = result['n.npi'], result['n.id']
                    fullname = result.get('n.fullname') or f"{result.get('n.fname', '')} {result.get('n.lname', '')}".strip()

                    if not fullname:
                        print(f"Missing name for NPI {npi}, {id}")
                        continue

                    registry_results = self.check_against_gov_registry(npi)
                    if not registry_results:
                        print(f'No data returned for NPI {npi}, {id}')
                        continue

                    for reg in registry_results:
                        reg_name = f"{reg['basic'].get('first_name', '')} {reg['basic'].get('last_name', '')}".strip()
                        ratio = fuzz.ratio(fullname.lower(), reg_name.lower())
                        if ratio != 100:
                            output_f.write(f'{fullname} vs {reg_name} : {ratio} for NPI {npi}, {id}\n')
