from abc import abstractmethod
from sqlalchemy import create_engine
import pandas as pd

from utils import auto_config as config
from sqlalchemy.orm import sessionmaker
from abc import abstractmethod
from urllib.parse import quote
import numpy as np 

class DatabaseManager:
    def __init__(self, engine_info):
        self.engine_info = engine_info

    @abstractmethod
    def connect(self):
        pass

class DatabaseFactory:
    class DatabaseType:
        SQLITE = "sqlite"
        POSTGRES = "postgres"
        MYSQL = "mysql"
        REDIS = "redis"
        ARANGO = "arango"
        LN = "lightning_network"

    @classmethod
    def build_database_manager(cls, type):
        """
        Initialize database manager.
        Args:
            type: type of database eg: sqlite, postgres, etc.

        Returns:

        """

        database_manager = None
        DatabaseFactory.database_type = type
        if type == DatabaseFactory.DatabaseType.SQLITE:
            engine_info = (
                f"sqlite:///{config.SQLITE_PATH}"
                if hasattr(config, "SQLITE_PATH")
                else "sqlite:///sql_app.db"
            )
            database_manager = PostgresDatabaseManagerImpl(engine_info)
        elif type == DatabaseFactory.DatabaseType.POSTGRES:
            host, port, database, user, password = (
                config.DB_HOST,
                config.DB_PORT,
                config.DB_DATABASE,
                config.DB_USER,
                config.DB_PASSWORD,
            )
            prefix = "postgresql"
            engine_info = (
                f"{prefix}://{user}:{quote(password)}@{host}:{port}/{database}"
            )
            database_manager = PostgresDatabaseManagerImpl(engine_info)
            database_manager.connect(True if config.DEBUG else False)
        elif type == DatabaseFactory.DatabaseType.MYSQL:
            host, port, database, user, password = (
                config.DB_HOST,
                config.DB_PORT,
                config.DB_DATABASE,
                config.DB_USER,
                config.DB_PASSWORD,
            )
            prefix = "mysql+pymysql"

            engine_info = (
                f"{prefix}://{user}:{quote(password)}@{host}:{port}/{database}"
            )
            database_manager = PostgresDatabaseManagerImpl(engine_info)
            database_manager.connect()
        else:
            raise Exception(f"Database {type} has not been supported yet.")

        return database_manager
        

class PostgresDatabaseManagerImpl(DatabaseManager):
    def connect(self, show_sql_query_flag=False):
        self.engine = create_engine(self.engine_info, echo=show_sql_query_flag)
        session = sessionmaker(bind=self.engine)
        self.session = session()
        

    def to_dataframe(self, sql, chunksize=None):
        data = pd.read_sql_query(sql, con=self.engine, chunksize=chunksize)
        return data

    def add_limit_if_debug(self, sql_command):
        new_query = f"{sql_command} limit {config.DB_LIMIT_QUERY_RECORDS}" if config.DEBUG else sql_command
        return new_query

    def execute(self, sql):
        result = self.session.execute(sql)
        self.session.commit()
        self.session.flush()
        return result

    def get_column_names(self, table_name,  schema = "public"):
        sql = f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema}' AND table_name   = '{table_name}';
        """
        return self.execute(sql=sql).all()
    
    def get_column_index(self, column_name, table_name, schema = "public"):
        columns = self.get_column_names(table_name=table_name, schema=schema)
        index_array = np.argwhere(np.array(columns) == column_name)
        return int(index_array[0][0])
