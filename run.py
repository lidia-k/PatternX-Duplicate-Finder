import argparse

#from DF_adventureworks.duplicate_finder import DuplicateFinder
from DF_penumbra.data_processor import DataProcessor


if __name__ == '__main__':
    choices = ['adventureworks', 'penumbra']

    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument('project', choices=choices, type=str, help='The project to run')
    args = parser.parse_args()

    if args.project == choices[0]:
        print(f'Running it for {choices[0]}')

        TABLE_NAME = 'production.product'
        duplicate_finder = DuplicateFinder('sentence-transformers/all-MiniLM-L6-v2', TABLE_NAME)
        #duplicate_finder.extract_lowest_distances(English=True)

        size_prompt = """The product subcategory is the most important feature to consider.
        The difference in size is less important than the product subcategory.
        """
        #For example, the size 42 is very different from 48."""
        example1 = [964, 965, 961]
        example2 = [765, 766, 768]

        duplicate_finder.extract_distance_between_pairs([765, 10001, 10002], prompt=size_prompt, English=True)
    
    elif args.project == choices[1]:
        print(f'Running it for {choices[1]}')
        DataProcessor().import_csv_to_neo4j()