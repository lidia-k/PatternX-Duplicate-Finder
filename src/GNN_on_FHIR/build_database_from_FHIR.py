# Imports needed
import glob
import json
import os

# Imports from other local python files
from NEO4J_Graph import Graph
from FHIR_to_graph import resource_to_node, resource_to_edges


NEO4J_URI = os.getenv('NEO4J_URL')
USERNAME = os.getenv('NEO4J_USER')
PASSWORD = os.getenv('NEO4J_PASSWORD')

graph = Graph(NEO4J_URI, USERNAME, PASSWORD)

print(graph.resource_metrics())
print(graph.database_metrics())
graph.wipe_database()

synthea_bundles = [glob.glob("../../synthea_sample_data_fhir_latest/*.json")[0]]
synthea_bundles.sort()
print(f"file name: {synthea_bundles}")
print(f"Number of files: {len(synthea_bundles)}")

nodes = []
edges = []
dates = set() # set is used here to make sure dates are unique
for bundle_file_name in synthea_bundles:
    with open(bundle_file_name) as raw:
        bundle = json.load(raw)
        for entry in bundle['entry']:
            resource_type = entry['resource']['resourceType']
            if resource_type != 'Provenance':
                # generated the cypher for creating the resource node 
                nodes.append(resource_to_node(entry['resource']))
                # generated the cypher for creating the reference & date edges and capture dates
                node_edges, node_dates = resource_to_edges(entry['resource'])
                edges += node_edges
                dates.update(node_dates)

# create the nodes for resources
for node in nodes:
    graph.query(node)

# create the nodes for dates
for date in dates:
    cypher = 'CREATE (:Date {name:"' + date + '", id: "' + date + '"})'
    graph.query(cypher)

# create the edges
for edge in edges:
    try:
        graph.query(edge)
    except:
        print(f'Failed to create edge: {edge}')

