import pandas as pd 
import typing as tp 
import numpy as np 
pd.options.mode.copy_on_write = True

class DataPreprocessor:
    def preprocess_data(self, data):
        pass


class PenumbraDataPreprocessor(DataPreprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.dtype_dict = {
            'Status': str,
            'Contact Type': str,
            'First Name': str,
            'Last Name': str,
            'Full Name': str,
            'HCP Category': str,
            'Payments Made To:': str,
            'SAP Number': str,
            'SAP Entity Name': str,
            'State/Region/Province': str,
            'Country': str,
            'National Physician ID': str,
            'Unnamed': np.float64,
            'Specialty': str,
            'Payment Currency':str,
            'Email Address': str,
            'Quickbase Record ID#': np.float64,
            'Reportable HCP': 'str',
            'National Physician ID/RPPS ID': str,
            'Primary Organization': str,
            'Primary Organization Type': str
        }

    def preprocess_data(self, data: tp.Tuple[pd.DataFrame, pd.DataFrame]):
        A, B = data 

        # rename special column
        A = A.rename(columns={'Unnamed: 14': "Unnamed"})
        B = B.rename(columns={'Unnamed: 15': "Unnamed"})

        # Normalize NPI data 
        A = self.normalize_NPI(A)
        B = self.normalize_NPI(B)

        # remmoved missing feature data
        removed_features_for_A = self.detect_high_missing_features(A, missing_percentage_threshold=59.2)
        removed_features_for_B = self.detect_high_missing_features(B, missing_percentage_threshold=59.2)
        removed_features = set(removed_features_for_A) & set(removed_features_for_B)
        A = A[[col for col in A.columns if col not in removed_features]]
        B = B[[col for col in B.columns if col not in removed_features]]

        # only using the common features between A and B
        common_columns = [col for col in A.columns if col in B.columns]
        A = A[common_columns]
        B = B[common_columns]

        # filling default value for empy/Null values. 
        missing_columns_for_A = self.detect_high_missing_features(A, missing_percentage_threshold=0)
        missing_columns_for_B = self.detect_high_missing_features(B, missing_percentage_threshold=0)

        for col in missing_columns_for_A:
            A = self.fill_missing_data(A, col)

        for col in missing_columns_for_B:
            B = self.fill_missing_data(B, col)

        # process for SAP Number
        special_SAP_df = A[A['SAP Number'].astype(str).str.contains('\n')].groupby("SAP Number").apply(self.process_biSAP_number).reset_index(drop = True)
        A = A[~A['SAP Number'].astype(str).str.contains('\n')]
        A = pd.concat([A, special_SAP_df])

        special_SAP_df = B[B['SAP Number'].astype(str).str.contains('\n')].groupby("SAP Number").apply(self.process_biSAP_number).reset_index(drop = True)
        B = B[~B['SAP Number'].astype(str).str.contains('\n')]
        B = pd.concat([B, special_SAP_df])

        A["SAP Number"] = A["SAP Number"].astype(str) 
        B["SAP Number"] = B["SAP Number"].astype(str) 

        return A, B 


    
    def detect_high_missing_features(self, df: pd.DataFrame, missing_percentage_threshold: float = 30) -> tp.List[str]:
        """Identify features in dataframe that have a lots of missing values. 
        We use a percentage threshold to capture feature that has more than that and 
        list all of them. 

        Args:
            df (pd.DataFrame): input data as DataFrame
            missing_percentage_threshold (float, optional): threshold to identify a significant missing value feature. Defaults to 30.

        Returns:
            List: a list of significant missing value features
        """
        missing_percentage = df.isna().sum()/df.shape[0] * 100
        removed_missed_features = missing_percentage[missing_percentage > missing_percentage_threshold].index
        return removed_missed_features
    
    def normalize_NPI(self, df: pd.DataFrame):
        df['National Physician ID'] = df['National Physician ID'].fillna('UNKNOWN').astype(str)
        return df 
    
    def process_biSAP_number(self, df):
        df_1 = df.copy()
        df['SAP Number'] = df['SAP Number'].map(lambda x: x.split("\n")[0])
        df_1['SAP Number'] = df_1['SAP Number'].map(lambda x: x.split("\n")[1])

        df['SAP Entity Name'] = df['SAP Entity Name'].map(lambda x: x.split("\n")[0])
        df_1['SAP Entity Name'] = df_1['SAP Entity Name'].map(lambda x: x.split("\n")[1])

        return pd.concat([df, df_1]) 

    def fill_missing_data(self, df, column):
        default = None
        type = self.dtype_dict[column]

        if type == 'datetime64[ns]':
            df[column] = pd.to_datetime(df[column])
            return df 

        if (type == str):
            default = "UNKNOWN" 
        elif type == np.int32:
            default = -1
        elif type == np.float64:
            default = -1
        elif type == object:
            default = "UNKNOWN"
        else: 
            default = "UNKNOWN"

        df[column] = df[column].fillna(default).astype(type)

        return df 
