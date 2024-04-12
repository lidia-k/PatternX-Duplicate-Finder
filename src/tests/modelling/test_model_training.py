from src.data.data_collection import PenumbraDataCollector
from src.preprocessing.data_preprocessing import PenumbraDataPreprocessor
from src.preprocessing.feature_engineering import PenumbraFeatureEnginner
from src.modelling.model_trainining import PenumbraModelTrainer


def test_PenumbraModelTrainer():
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
    assert (model != None)
