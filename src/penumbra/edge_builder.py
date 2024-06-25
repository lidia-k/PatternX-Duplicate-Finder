import pandas as pd
from collections import defaultdict

from src.dao.NEO4J_Graph import Graph, VectorGraph
from src.penumbra import constants
from src.utils import auto_config as config


class EdgeBuilder:
    def __init__(self):
        self.graph = Graph(
            config.NEO4J_URL,
            config.NEO4J_USER,
            config.NEO4J_PASSWORD,
        )
        self.distinct_pairs = set()
        self.uids = []

    def _build_o_dup_edges(self, session):
        """
        If two nodes have the same property, create an edge between them.
        We call these obvious duplicates.
        """
        for type in constants.EDGE_TYPES:
            if type == "npi":
                q = f"""
                    MATCH (a),(b)
                    WHERE a.{type} = b.{type} AND id(a) < id(b)
                    MERGE (a)-[:r1_{type}]-(b)
                    RETURN count(*)
                    """
            else:
                p1, p2 = type.split("_", 1)
                q = f"""
                    MATCH (a),(b)
                    WHERE a.{p1} = b.{p1} AND a.{p2} = b.{p2} AND id(a) < id(b)
                    MERGE (a)-[:r1_{type}]-(b)
                    RETURN count(*)
                    """
            result = session.run(q)
            print(f"Built {result.single()[0]} edges for {type}")

    def _create_gds_graph(self, session):
        """
        If the graph exists, drop it and create a new one.
        GDS graph is used to find connected clusters.
        """
        check_q = "CALL gds.graph.exists('penumbra') YIELD exists RETURN exists"
        result = session.run(check_q).data()

        if result[0]["exists"]:
            drop_q = "CALL gds.graph.drop('penumbra')"
            session.run(drop_q)

        build_q = "CALL gds.graph.project('penumbra', '*', '*')"
        session.run(build_q)

    def _fetch_clustsers(self, session):
        stream_q = """
        CALL gds.wcc.stream('penumbra') YIELD nodeId, componentId 
        RETURN gds.util.asNode(nodeId).uid AS name, componentId 
        ORDER BY componentId, name
        """
        result = session.run(stream_q).data()

        clusters = defaultdict(list)
        for record in result:
            clusters[record["componentId"]].append(record["name"])

        return clusters

    def _fetch_clusters_by_label(self, session):
        labels = ["Provider", "Speaker"]
        clusters = defaultdict(lambda: defaultdict(list))

        for label in labels:
            stream_q = f"""
            MATCH (n:{label})
            CALL gds.wcc.stream('penumbra') YIELD nodeId, componentId 
            WHERE id(n) = nodeId
            RETURN gds.util.asNode(nodeId).uid AS name, componentId 
            ORDER BY componentId, name
            """
            result = session.run(stream_q).data()
            
            for record in result:
                clusters[label][record["componentId"]].append(record["name"])

        return clusters
    
    def _create_m_node_and_relationship(self, session, uids, i):
        # Fetch all the nodes in the cluster
        q = """
        MATCH (n)
        WHERE n.uid IN $uids
        RETURN n
        """
        result = session.run(q, uids=uids).data()

        # Aggregate the properties of the nodes
        master_props = {}
        master_props["uid"] = f"r1_m_{i}"
        for node in result:
            for key, value in node["n"].items():
                if key in ["uid", "text", "embedding"]:
                    continue
                if not master_props.get(key):
                    master_props[key] = value
                elif master_props[key] != value:
                    master_props[key] = f"{master_props[key]}, or {value}"

        # Create a master node with the aggregated properties
        create_q = """
        CREATE (m:Master $master_prop)
        RETURN m
        """
        m_node = session.run(create_q, master_prop=master_props).single()[0]

        # Set the text property for the master node
        # text, q = DataProcessor._build_text(m_node)
        # session.run(q, uid=m_node['uid'], text=text)

        # Create a relationship between the master node and all the nodes in the cluster
        for id in uids:
            q = """
            MATCH (m:Master), (n) WHERE m.uid = $m_uid AND n.uid = $n_uid
            MERGE (m)-[:r1_master]-(n)
            """
            session.run(q, m_uid=m_node["uid"], n_uid=id)

    def count_distinct_clusters_by_label(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            clusters = self._fetch_clusters_by_label(session)

            unique_entities = {}
            for label, cluster_data in clusters.items():
                unique_entities[label] = len(cluster_data.values())
                # for uids in cluster_data.values():
                #     # If the cluster has more than one node, create a master node
                #     if len(uids) > 1:
                #         unique_entities[label] += 1
            print(unique_entities)

    def _create_r1_master_nodes(self, session):
        self._create_gds_graph(session)
        clusters = self._fetch_clustsers(session)
        print(f"Found {len(clusters)} clusters")

        i = 1
        for uids in clusters.values():
            # If the cluster has more than one node, create a master node
            if len(uids) > 1:
                self._create_m_node_and_relationship(session, uids, i)
                i += 1
        print(f"Created {i-1} master nodes")

    def handle_master(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            self._create_r1_master_nodes(session)
            
    def handle_o_dups(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            self._build_o_dup_edges(session)
            # self._create_r1_master_nodes(session)

    def lookup_o_dups(self):
        driver = self.graph.get_driver()
        with driver.session() as session:
            for type in constants.EDGE_TYPES:
                q = f"""
                    MATCH (a)-[:r1_{type}]-(b)
                    WHERE id(a) < id(b)
                    RETURN a, b
                    """
                result = session.run(q).data()
                print(f"Found {len(result)} duplicates for {type}")

    def _fetch_nodes(self):
        q = """
        MATCH (master:Master)-[:r1_master]-(n)
        RETURN master AS node

        UNION

        MATCH (n)
        WHERE NOT (n)-[:r1_master]-() AND NOT (n:Master)
        RETURN n AS node
        """
        return self.graph.cypher_transaction(q)

    def _process_similar_nodes(self, similar_nodes, uid, results_list):
        for s_node in similar_nodes:
            doc, score = s_node
            metadata = doc.metadata
            doc_id = metadata.get("uid")

            # If the node is not a master node or an unconnected node, skip it
            if doc_id not in self.uids:
                continue

            pair = tuple(sorted([uid, doc_id]))

            # Return the nodes that have a score greater than 0.96, aren't themselves, and are distinct pairs.
            if score > 0.96 and uid != doc_id and pair not in self.distinct_pairs:
                self.distinct_pairs.add(pair)
                doc_dict = {}
                doc_dict = {"score": score}
                doc_dict.update(
                    {col: metadata.get(col, "") for col in constants.COLS_TO_USE}
                )
                results_list.append(doc_dict)

        return results_list

    def similarity_search(self):
        # Generate embeddings for the text properties of all nodes
        for node in ["Provider", "Speaker", "Master"]:
            vector_g = VectorGraph(node)

        csv_data = []
        empty_row = {col: "" for col in constants.COLS_TO_USE}

        # Fetch master nodes and nodes that are not connected to master nodes
        result = self._fetch_nodes()
        self.uids = [record[0]["uid"] for record in result]

        for record in result:
            node = record[0]
            uid = node["uid"]
            text = node["text"]

            # The original node is the first row in the csv
            node_dict = {col: node.get(col, "") for col in constants.COLS_TO_USE}
            node_dict["score"] = ""

            results_list = []
            # Do similarity search and process the results
            similar_nodes = vector_g.similarity_search_with_score(text, k=5)
            self._process_similar_nodes(similar_nodes, uid, results_list)

            # If there are similar nodes, add them to the csv
            if results_list:
                csv_data.append(empty_row)
                csv_data.append(node_dict)
                csv_data.extend(results_list)

        # Write the results to a csv file
        df = pd.DataFrame(csv_data)
        df.fillna("", inplace=True)
        df.to_csv("similarity_search.csv", index=False)

    def set_relationship(self, df: pd.DataFrame, relationship: str):
        if not ("ltable_uid" in df.columns) and not ("rtable_uid" in df.columns):
            raise "DataFrame is missing ltable_uid and rtable_uid columns"

        for index, row in df.iterrows():
            # print(index, row)
            q = f"""
            MATCH (l), (r)
            WHERE l.uid = "{row["ltable_uid"]}" AND r.uid = "{row["rtable_uid"]}"
            CREATE (l)-[:{relationship}]->(r)
            """
            self.graph.cypher_transaction(q)

        print(
            "DONE - created relationship '{}' for {} pairs".format(
                relationship, len(df)
            )
        )
