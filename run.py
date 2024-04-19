import argparse

#from DF_adventureworks.duplicate_finder import DuplicateFinder
from src.modelling.model_deployment import PenumbraModelDeployer
from src.data.data_collection import PenumbraDataCollector
from src.modelling.model_trainining import PenumbraModelTrainer
from src.preprocessing.data_preprocessing import PenumbraDataPreprocessor
from src.preprocessing.feature_engineering import PenumbraFeatureEnginner
from src.DF_penumbra.data_processor import DataProcessor
#from src.DF_penumbra.npi_vaildator import NPIValidator
from src.DF_penumbra.duplicate_finder import DuplicateFinder
import src.utils.auto_config as config 
import pandas as pd 

if __name__ == '__main__':
    choices = ['adventureworks', 'penumbra']

    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument('--project', choices=choices, type=str, help='The project to run')
    parser.add_argument("--task", type=str, default=None, help="task name:{train, predict, online_train}",  metavar='')
    parser.add_argument("--model", type=str, default=None, help="model name",  metavar='')
    parser.add_argument("--data", type=str, default=None, help="data file name",  metavar='')

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
        print(f"model name: {args.model} data file: {args.data}  online training...")
        model = PenumbraModelDeployer.__new__(PenumbraModelDeployer).deploy_model(config.MODEL_FOLDER + args.model).model
        data_file = config.DATA_DIR + args.data
        
        trainer = PenumbraModelTrainer()
        trainer.online_training(model, data_file)
    elif args.project == 'penumbra' and args.task == 'predict':
        print(f"model name: {args.model} data file: {args.data} predict...") # "demo_model.pth"
        
        file_path = config.DATA_DIR + args.data
        df = pd.read_csv(file_path)
        if "_id" not in df.columns.tolist():
            df = df.reset_index(drop=True).reset_index().rename(columns = {"index": "_id"})
            df.to_csv(file_path)
        
        predictions = PenumbraModelDeployer.__new__(PenumbraModelDeployer).deploy_model(config.MODEL_FOLDER +args.model).predict(file_path)
        selected_columns = [
            'ltable_Full Name',
            'rtable_Full Name',
            'ltable_Email Address',
            'rtable_Email Address',
            # 'ltable_National Physician ID',
            # 'rtable_National Physician ID',
            'confident', 'is_matched',
            'label']

        print("\nsample predictions")
        print(predictions[selected_columns].head(10))
        print("\nWrong predictions")
        wrong_predictions = predictions.query("(is_matched == 0) & (label == 1) ").head(20).reset_index()
        print(wrong_predictions[selected_columns])
        
        write_path = config.DATA_DIR + "/wrong_prediction.csv"
        print(f"Writing wrong prediction into: {write_path} ")
        
        online_learning_selected_columns = ['ltable_Email Address', 'ltable_First Name',
       'ltable_Quickbase Record ID#', 'ltable_Specialty', 'ltable_Last Name',
       'ltable_Status', 'ltable_SAP Number', 'ltable_Payments Made To:',
       'ltable_Payment Currency', 'ltable_HCP Category',
       'ltable_State/Region/Province', 'ltable_Full Name',
       'ltable_Contact Type', 'ltable_National Physician ID', 'ltable_Country',
       'ltable_SAP Entity Name', 'rtable_Email Address', 'rtable_First Name',
       'rtable_Quickbase Record ID#', 'rtable_Specialty', 'rtable_Last Name',
       'rtable_Status', 'rtable_SAP Number', 'rtable_Payments Made To:',
       'rtable_Payment Currency', 'rtable_HCP Category',
       'rtable_State/Region/Province', 'rtable_Full Name',
       'rtable_Contact Type', 'rtable_National Physician ID', 'rtable_Country',
       'rtable_SAP Entity Name', 'label']
        
        wrong_predictions[online_learning_selected_columns].to_csv(write_path, index = False)
        
    elif args.project == 'penumbra':
        print(f'Running it for {choices[1]}')
        
        dp = DataProcessor(data_dir='src/data')
        #dp.import_csv_to_neo4j()
        #dp.add_text_props()
        #dp.detect_high_missing_features()
        dp.build_matching_pairs()

        #NPIValidator().validate_NPIs()
        
        df = DuplicateFinder()
        #df.process_o_dups() # Obvious duplicates
        #df.lookup_o_dups()
        #df.similarity_search()

