from src.data.data_collection import PenumbraDataCollector
from src.preprocessing.data_preprocessing import PenumbraDataPreprocessor


def test_PenumbraDataPreprocessor():
    data_dir = '/Users/tu/SourceCode/notebooks/data/'
    file = "hcp-manz-sn.xlsx"
    p = PenumbraDataCollector(data_dir, file)
    p.collect_data()

    df_dict = p.df_dict

    A, B = p.build_data_pair(
        items_in_A=[df_dict['(1000) Contacts']], 
        items_in_B=[df_dict['(800) No SAP Number and Export '], df_dict['(340) US HCPs'], df_dict['(320) OUS HCPs'], df_dict['(20) France HCPs']]
    )

    p =  PenumbraDataPreprocessor()
    A, B = p.preprocess_data(data = (A, B) )

    assert (not A.empty ) & (not B.empty)
