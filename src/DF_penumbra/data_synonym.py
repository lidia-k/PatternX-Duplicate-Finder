import pandas as pd

from src.dao.NEO4J_Graph import Graph
from src.DF_penumbra.utils import get_synonyms
from src.utils import auto_config as config


class Neo4jSynonymNodeCreator:
    def __init__(self):
        self.graph = Graph(config.NEO4J_URL, config.NEO4J_USER, config.NEO4J_PASSWORD)

    def create_synonym_nodes(self):
        synonym_dict = get_synonyms()
        fnames = list(synonym_dict.keys())
        q = f"MATCH (n) WHERE n.fname IN {fnames} RETURN n, id(n) AS node_id"
        result = self.graph.cypher_transaction(q)
        df = pd.DataFrame([dict(record[0], node_id=record[1]) for record in result])
        print(df)
        count = 0
        for index, row in df.iterrows():
            fname = row["fname"]
            synonyms = [s.strip() for s in synonym_dict[fname].split(",")]

            # Create synonym nodes and copy properties and relationships
            for synonym in synonyms:
                count += 1
                # Create the new node
                create_node_query = """
                MATCH (n)
                WHERE ID(n) = $original_id
                CREATE (copy:Synonym)
                SET copy = n
                SET copy.fname = $synonym
                SET copy.fullname = $fullname
                RETURN id(copy)"""
                result = self.graph.cypher_transaction(
                    create_node_query,
                    {
                        "original_id": row["node_id"],
                        "synonym": synonym,
                        "fullname": "{} {}".format(synonym, row["lname"]),
                    },
                )
                synonym_node_id = result[0][0]

                # Create the 'clone' relationship
                create_clone_rel_query = """
                MATCH (original), (synonym)
                WHERE id(original) = $original_node_id AND id(synonym) = $synonym_node_id
                CREATE (original)-[:r_clone]->(synonym)
                """

                self.graph.cypher_transaction(
                    create_clone_rel_query,
                    {
                        "original_node_id": row["node_id"],
                        "synonym_node_id": synonym_node_id,
                    },
                )

                # Copy relationships from original node to new synonym node
                self._copy_relationships(row["node_id"], synonym_node_id)
        print(f"Added node {count}")

    def _copy_relationships(self, original_node_id, synonym_node_id):
        copy_outgoing_rels_query = """
        MATCH (n)-[r]->(m)
        WHERE id(n) = $original_node_id AND type(r) <> 'r_clone'
        WITH type(r) AS relType, m
        MATCH (synonym)
        WHERE id(synonym) = $synonym_node_id
        CALL apoc.create.relationship(synonym, relType, {}, m) YIELD rel
        RETURN synonym, m
        """
        self.graph.cypher_transaction(
            copy_outgoing_rels_query,
            {"original_node_id": original_node_id, "synonym_node_id": synonym_node_id},
        )

        copy_incoming_rels_query = """
        MATCH (n)<-[r]-(m)
        WHERE id(n) = $original_node_id AND type(r) <> 'r_clone'
        WITH type(r) AS relType, m
        MATCH (synonym)
        WHERE id(synonym) = $synonym_node_id
        CALL apoc.create.relationship(m, relType, {}, synonym) YIELD rel
        RETURN synonym, m
        """
        self.graph.cypher_transaction(
            copy_incoming_rels_query,
            {"original_node_id": original_node_id, "synonym_node_id": synonym_node_id},
        )
