import argparse
from src.DF_penumbra.data_loader import Neo4jDataLoader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument("--file", type=str, default="master_group.csv", help="filepath",  metavar='')

    args = parser.parse_args()
    dl = Neo4jDataLoader(data_dir=None)
    pair_nodes = dl.import_human_labeled(args.file)
    print(f"X pairs: {len(pair_nodes)}")
