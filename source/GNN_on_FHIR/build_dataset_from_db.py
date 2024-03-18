import os
import pickle

from NEO4J_Graph import Graph


NEO4J_URI = os.getenv('NEO4J_URL')
USERNAME = os.getenv('NEO4J_USER')
PASSWORD = os.getenv('NEO4J_PASSWORD')

graph = Graph(NEO4J_URI, USERNAME, PASSWORD)

target_dir = './preprocessed_datapoints'
os.makedirs(target_dir, exist_ok=True)

driver = graph.get_driver()
with driver.session() as session:
    datapoint_ids = session.run('MATCH (p:Patient) RETURN p.id').value()
    
    prop_query = """
    MATCH (n)
    WITH labels(n) AS types, keys(n) AS props
    UNWIND types AS type
    RETURN DISTINCT type, collect(DISTINCT props) AS properties
    """
    prop_results = session.run(prop_query)
    
    num = 0
    node_type_to_int = {}
    features = {}
    for resource in prop_results:
        resource_type = resource['type']
        if resource_type == 'resource':
            continue

        features[resource_type] = {}
        properties = set([prop for sublist in resource['properties'] for prop in sublist])
        for prop in properties:
            features[resource_type][prop] = []
        
        node_type_to_int[resource_type] = num 
        num += 1 

edge_type_to_int = {}
edge_num = 0 
base_query = """
MATCH (p:Patient {id: $id})-[r]-(connectedNode)
OPTIONAL MATCH (connectedNode)-[r2]-(otherConnectedNode)
WHERE id(connectedNode) < id(otherConnectedNode)
RETURN p, collect(DISTINCT connectedNode) as ConnectedNodes, 
       collect(DISTINCT r) + collect(DISTINCT r2) as Relationships
"""
for i in range(3):
    with driver.session() as session:
        result = session.run(base_query, id=datapoint_ids[i]).single()
        p_node = result['p']
        connected_nodes = result['ConnectedNodes']
        relationships = result['Relationships']

        all_nodes = [p_node] + connected_nodes
        neo4j_id_to_graph_idx = {node.element_id: idx for idx, node in enumerate(all_nodes)}
        node_types = [None] * len(all_nodes)
        for node in all_nodes:
            node_type = tuple(node.labels)[1] if tuple(node.labels)[0] == 'resource' else tuple(node.labels)[0]
            node_idx = neo4j_id_to_graph_idx[node.element_id]
            node_types[node_idx] = node_type_to_int[node_type]

            for feature_name, feature_values in features[node_type].items():
                value = node.get(feature_name)
                feature_values.append(value)

        edge_list = []
        edge_types = []
        rel_type = []
        for rel in relationships:
            snode_labels = rel.start_node.labels
            if bool(snode_labels):
                snode_type = tuple(snode_labels)[1] if tuple(snode_labels)[0] == 'resource' else tuple(snode_labels)[0]
            else: 
                snode_type = 'Date'

            enode_labels = rel.end_node.labels
            if bool(enode_labels):
                enode_type = tuple(enode_labels)[1] if tuple(enode_labels)[0] == 'resource' else tuple(enode_labels)[0]
            else: 
                enode_type = 'Date'

            edge_type = f'{snode_type}_To_{enode_type}'
            if edge_type not in edge_type_to_int.keys():
                edge_type_to_int[edge_type] = edge_num
                edge_num += 1

            snode_idx = neo4j_id_to_graph_idx[rel.start_node.element_id]
            enode_idx = neo4j_id_to_graph_idx[rel.end_node.element_id]
            edge_list.append((snode_idx, enode_idx))
            edge_types.append(edge_type_to_int[edge_type])

        with open(os.path.join(target_dir, str(datapoint_ids[i])), 'wb') as f:
            dp_tuple = (edge_list, node_types, edge_types, features)
            pickle.dump(dp_tuple, f)
 


