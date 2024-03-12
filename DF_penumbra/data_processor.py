import glob
import psycopg2

from utils import auto_config as config


class DataProcessor:
    def __init__(self):
        self._connect_to_db()

    def _connect_to_db(self):
        try: 
            conn = psycopg2.connect(
                dbname=config.DB_DATABASE,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT
            )
            cur = conn.cursor()
            return conn, cur 
        except Exception as e:
            print(f'Failed to connect to the db: {e}')
            return 

    def insert_data(self):
        data_bundles = glob.log("../../*.csv")
