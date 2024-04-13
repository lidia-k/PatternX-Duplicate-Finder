import typing as tp 
import pandas as pd 
import msoffcrypto
import src.utils.auto_config as config 
import io 

class DataCollector:
    def collect_data(self):
        pass


class PenumbraDataCollector(DataCollector):
    def __init__(self, data_dir, data_file_name="hcp-manz-sn.xlsx") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.data_file_name = data_file_name
        self.df_dict = {}

    def collect_data(self):
        decrypted_workbook = io.BytesIO()
        with open(self.data_dir + self.data_file_name , 'rb') as file:
            office_file = msoffcrypto.OfficeFile(file)
            office_file.load_key(password=config.DATA_PASSWORD)
            office_file.decrypt(decrypted_workbook)

        for sheet_name in pd.read_excel(decrypted_workbook, None).keys():
            df = pd.read_excel(decrypted_workbook, sheet_name=sheet_name)
            self.df_dict[sheet_name] = df
    
    def build_data_pair(self, items_in_A: tp.List[ tp.Union[str, pd.DataFrame]], items_in_B: tp.List[ tp.Union[str, pd.DataFrame]]) -> tp.Tuple[pd.DataFrame]:
        """Building pair of data. One pair contains dataset A and dataset B. Two datasets will be used in Entity Matching. 

        Args:
            items_in_A (tp.List[ tp.Union[str, pd.DataFrame]]): Items in the first dataset A.
            items_in_B (tp.List[ tp.Union[str, pd.DataFrame]]): Items in the first dataset B.

        Returns:
            tp.Tuple[pd.DataFrame]: A pair of data that are used to Entity Matching. 
        """
    
        A_items = self.process_items(items_in_A)
        B_items = self.process_items(items_in_B)

        return (pd.concat(A_items), pd.concat(B_items))
    
    def process_items(self, items:tp.Union[str, pd.DataFrame]) -> tp.List[pd.DataFrame]:
        """Process items for a specified dataset. 

        Args:
            items (tp.Union[str, pd.DataFrame]): items of dataset. It could be a path to data or a dataframe. 

        Raises:
            Exception: Exception when we pass wrong data type of items. We may support relational table in near future as well. 

        Returns:
            tp.List[pd.DataFrame]: list of items in a dataset. 
        """
        item_dfs = []
        for item in items:
            if isinstance(item, str):
                item_df = pd.read_csv(item)
                item_dfs.append(item_df)
            elif isinstance(item, pd.DataFrame):
                item_dfs.append(item)
            else:
                raise Exception("Data type is not supported!")
            
        return item_dfs