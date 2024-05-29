import argparse
import joblib

import pandas as pd 
import numpy as np

#from DF_adventureworks.duplicate_finder import DuplicateFinder
from src.modelling.model_evaluation import PenumbraEvaluation
from src.modelling.model_deployment import PenumbraModelDeployer
from src.data.data_collection import PenumbraDataCollector
from src.modelling.model_trainining import PenumbraModelTrainer
from src.preprocessing.data_preprocessing import PenumbraDataPreprocessor
from src.preprocessing.feature_engineering import PenumbraFeatureEnginner
from src.DF_penumbra.data_loader import Neo4jDataLoader
from src.DF_penumbra.data_preprocessor import DataPreprocessor
from src.DF_penumbra.edge_builder import EdgeBuilder
from src.DF_penumbra.npi_vaildator import NPIValidator
from src.DF_penumbra.training_magellan import MagellanTrainer
import src.utils.auto_config as config 
import src.lib.deepmatcher as dm


if __name__ == '__main__':
    choices = ['adventureworks', 'penumbra']
    model_choices = ['dt', 'svm', 'rf', 'lg', 'ln', 'nb', 'xgb']

    parser = argparse.ArgumentParser(description='Run different functions based on input parameters.')
    parser.add_argument('--project', choices=choices, type=str, help='The project to run')
    parser.add_argument("--task", type=str, default=None, help="task name:{train, predict, online_train}",  metavar='')
    parser.add_argument("--m_model", choices=model_choices, type=str, default=None, help="Magellan model name",  metavar='')
    parser.add_argument("--size", type=int, default=None, help="Size parameter for the task", metavar='')
    parser.add_argument("--npi", action="store_true", help="Include NPIs for training (default: exclude NPIs)")
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
    elif args.project == 'penumbra' and args.task == 'eval':
        print("Evaluating model...")
        train, validation, test = dm.data.process(
            path=config.DATA_DIR,
            train='train.csv',
            validation='valid.csv',
            test='test.csv',
            use_magellan_convention=True
        ) 

        model_evaluation = PenumbraEvaluation()
        model1 = model_evaluation.load_model(config.MODEL_FOLDER +"model.pth")
        model2 = model_evaluation.load_model(config.MODEL_FOLDER +"retrained_model.pth")
        
        model_evaluation.compare_models(model1, model2, test)
    elif args.project == 'penumbra' and args.task == 'neo4j': 
        """
        Load the data to Neo4j and build edges for obvious duplicates.
        """     
        data_dir = 'src/data'

        print('Loading data to Neo4j')
        dl = Neo4jDataLoader(data_dir)
        dl.load_csv_to_neo4j()

        print('Building edges and master nodes for obvious duplicates')
        eb = EdgeBuilder()
        eb.handle_o_dups()

    elif args.project == 'penumbra' and args.task == 'synonym':
        dl = Neo4jDataLoader()
        dl.create_synonym_nodes()

    elif args.project == 'penumbra' and args.task == 'synoname':
        dl = Neo4jDataLoader()
        dl.create_synoname_nodes()

    elif args.project == 'penumbra' and args.task == 'm_training':
        """
        Prepare the training data, train the Magellan model, and evaluate the model.

        Args:
        - The --npi flag is optional. If included, the training data will include NPIs. The default is to exclude NPIs.
        - The --m_model flag. Specify the model to train. The model options are: 
            Decision Tree (dt), Support Vector Machine (svm), Random Forest (rf), 
            Logistic Regression (lg), Linear Regression (ln), XGBoost(xgb) and Naive Bayes (nb).

        Output:
        - A model.pkl file will be saved.
        - A droped.csv file will be saved. 
        ('droppep.csv' contains the test data that has pairs with non-matching NPIs and isn't used for training.)
        """
        if not args.m_model:
            raise ValueError('Please specify the model to use for training.')
        
        data_dir = 'src/data'

        print('Preparing training data with{} NPI...'.format('' if args.npi else 'out'))
        dp = DataPreprocessor(data_dir, include_npi=args.npi, training=True)
        ltable, rtable, data = dp.prepare_training_data(skewed_factor=2, size=args.size, model=args.m_model)

        print('Training {} with{} NPI...'.format(args.m_model, '' if args.npi else 'out'))
        mt = MagellanTrainer(ltable, rtable, data, model=args.m_model, training=True)
        mt.train_model()

        print('Evaluating the model...')
        preds = mt.predict()
        mt.evaluate(preds)

        print('Displaying feature importance...')
        mt.retrieve_feature_importance()

    elif args.project == 'penumbra' and args.task == 'test1':  
        """
        Prerequisits: 
        - model.pkl file should be available from the training.
        - dropped.csv file should be available from the training.

        Run predictions on the test data and the same data with NPIs removed.
        """  
        if not args.m_model:
            raise ValueError('Please specify the model used for training.')

        print(f'Running the test 1 for {args.m_model}')

        data_dir = 'src/data'
        model = joblib.load('model.pkl')

        data = pd.read_csv('dropped.csv')
        data.drop(columns=['label'], inplace=True)
        
        # If the saved model is trained without NPIs, the test data is prepared without NPIs.
        include_npi = False
        if 'rtable_npi' in data.columns:
            include_npi = True

        dp = DataPreprocessor(data_dir, include_npi=include_npi)
        df = dp._handle_int_cols(data)
        if args.model != 'xgb':
            df = dp._impute_missing_features(df)
        A, B, C = dp._load_data(df)

        print('Running predictions on the label 0 test data with{} NPI...'.format('' if include_npi else 'out'))
        mt = MagellanTrainer(A, B, C, model)
        preds = mt.predict(C)
        print(f'False negatives: {(preds["predicted"] == 1).sum()} (out of {len(preds)} negative predictions)')
        
        if include_npi:     
            print('Running predictions on the label 0 test data with NPIs removed...')
            # mask npis 
            df['ltable_npi'] = np.nan
            df['rtable_npi'] = np.nan
            A, B, C = dp._load_data(data)
        
            mt = MagellanTrainer(A, B, C, model)
            preds = mt.predict(C)
            print(f'False negatives when NPIs are masked: {(preds["predicted"] == 1).sum()} (out of {len(preds)} negative predictions)')
    
    elif args.project == 'penumbra' and args.task == 'test2':
        if not args.m_model:
            raise ValueError('Please specify the model used for training.')
        
        print(f'Running the test 2 for {args.m_model}')

        data_dir = 'src/data'
        model = joblib.load('model.pkl')
        # If the model is trained with NPIs, make sure include --npi flag to the run command.
        dp = DataPreprocessor(data_dir, include_npi=args.npi)

        df = dp.prepare_test2_data()

        if args.m_model != 'xgb':
            df = dp._impute_missing_features(df)
        
        A, B, C = dp._load_data(df)

        mt = MagellanTrainer(A, B, C, model)
        preds = mt.predict(C)
        #mt.retrieve_feature_importance()

    elif args.project == 'penumbra' and args.task == 'predict-all':
        """
        Prerequisits:
        - model.pkl file should be available from the training.

        Run predictions on the entire data except for the training data
        """
        if not args.m_model:
            raise ValueError('Please specify the model used for training.')
        
        print(f'Running {args.m_model} over the entire data...')

        data_dir = 'src/data'
        model = joblib.load('model.pkl')
        dp = DataPreprocessor(data_dir, include_npi=args.npi)

        df = dp.prepare_all_data()
        if args.m_model != 'xgb':
            df = dp._impute_missing_features(df)
        print("df", df)

        A, B, C = dp._load_data(df)

        mt = MagellanTrainer(A, B, C, model)
        preds = mt.predict(C, all=True)

    elif args.project == 'penumbra' and args.task == 'feature':
        model = joblib.load('model.pkl')
        data_dir = 'src/data'

        mt = MagellanTrainer(model=model)
        mt.retrieve_feature_importance()
    
    elif args.project == 'penumbra' and args.task == 'npi':
        NPIValidator().validate_NPIs()

    
    

