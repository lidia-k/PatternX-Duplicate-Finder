import argparse

#from DF_adventureworks.duplicate_finder import DuplicateFinder
from src.data.data_collection import PenumbraDataCollector
from src.modelling.model_trainining import PenumbraModelTrainer
from src.preprocessing.data_preprocessing import PenumbraDataPreprocessor
from src.preprocessing.feature_engineering import PenumbraFeatureEnginner
from src.DF_penumbra.data_processor import DataProcessor
from src.DF_penumbra.npi_vaildator import NPIValidator
from src.DF_penumbra.duplicate_finder import DuplicateFinder

if __name__ == '__main__':
    choices = ['adventureworks', 'penumbra']

    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument('--project', choices=choices, type=str, help='The project to run')
    parser.add_argument("--task", type=str, default=None, help="task name:{train, predict, online_train}",  metavar='')

    args = parser.parse_args()

    if args.project == 'adventureworks':
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
    elif args.project == 'penumbra' and args.task == 'train':
        print("training...")
        data_dir = '/Users/tu/SourceCode/notebooks/data/'
        file = "hcp-manz-sn.xlsx"
        dc = PenumbraDataCollector(data_dir, file)
        dc.collect_data()

        df_dict = dc.df_dict

        A, B = dc.build_data_pair(
            items_in_A=[df_dict['(1000) Contacts']], 
            items_in_B=[df_dict['(800) No SAP Number and Export '], df_dict['(340) US HCPs'], df_dict['(320) OUS HCPs'], df_dict['(20) France HCPs']]
        )

        p =  PenumbraDataPreprocessor()
        A, B = p.preprocess_data(data = (A, B) ) 

        fe = PenumbraFeatureEnginner("blocking")
        df = fe.execute_strategy(A, B, fe.blocking_config)
        
        trainer = PenumbraModelTrainer()
        model = trainer.train_model(df)
        
    elif args.project == 'penumbra' and args.task == 'online_train':
        print("online training...")
    elif args.project == 'penumbra' and args.task == 'predict':
        print("predict...")
    elif args.project == 'penumbra':
        print(f'Running it for {choices[1]}')
        
        dp = DataProcessor()
        #dp.import_csv_to_neo4j()
        #dp.add_text_props()
        
        #NPIValidator().validate_NPIs()
        
        df = DuplicateFinder()
        #df.process_o_dups() # Obvious duplicates
        #df.lookup_o_dups()
        df.similarity_search()
