import json
import psycopg2
import xml.etree.ElementTree as ET
from utils import auto_config as config

class DataProcessor: 
    
    DB_NAME = 'Adventureworks'
    DB_USER = 'postgres'
    DB_PASSWORD = 'postgres'

    def __init__(self, table_name):
        self.table_name = table_name
        self.new_table = self.table_name + '_flattened'
        
        #self.create_duplicate_table()
        #self.update_catalog_description()
        #self.update_boolean_columns()
        #self.update_categorical_columns()
        #self.update_column_names()
        #self.update_foreignkey_columns()
        #self.drop_columns()
        self.update_weight_unit_measure_code()

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

    def create_duplicate_table(self):
        try:
            conn, cur = self._connect_to_db()
            cur.execute(f'CREATE TABLE {self.new_table} AS SELECT * FROM {self.table_name};')
            
            conn.commit()
            conn.close()
            print(f'Duplicate table of {self.table_name} successfully created')
        except Exception as e:
            print(f'Failed to create a dulicate table: {e}')

    def update_boolean_columns(self):
        try:
            conn, cur = self._connect_to_db()
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN MakeFlag TO Production_Type;')
            cur.execute(f'ALTER TABLE {self.new_table} ALTER COLUMN Production_Type TYPE VARCHAR;')
            
            update_query = f"""
            UPDATE {self.new_table}
            SET Production_Type = CASE 
                WHEN Production_Type = 'true' THEN 'Manufactured'  
                WHEN Production_Type = 'false' THEN 'Outsourced'  
            END;
            """
            cur.execute(update_query)

            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN FinishedGoodsFlag TO Product_Status;')
            cur.execute(f'ALTER TABLE {self.new_table} ALTER COLUMN Product_Status TYPE VARCHAR;')
            
            update_query = f"""
            UPDATE {self.new_table}
            SET Product_Status = CASE 
                WHEN Product_Status = 'true' THEN 'Ready for sale'  
                WHEN Product_Status = 'false' THEN 'Not ready for sale'  
            END;
            """
            cur.execute(update_query)

            conn.commit()
            conn.close()
            
            print('Boolean columns are successfully updated')
        except Exception as e:
            print(f'Failed to update the boolean columns: {e}')

    def update_column_names(self):
        try:
            conn, cur = self._connect_to_db()
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN SafetyStockLevel TO Miminum_Quantity_Stock_In_Inventory;')
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN ReorderPoint TO Reorder_Status_Indicator;')
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN StandardCost TO Estimated_Product_Cost_In_USD;')
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN ListPrice TO Suggested_Retail_Price_In_USD;')
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN DaysToManufacture TO Production_Lead_Time_In_Days;')

            conn.commit()
            conn.close()

            print('Column names are successfully updated')
        except Exception as e:
            print('Failed to update the column names: {e}')

    def update_categorical_columns(self):
        try: 
            conn, cur = self._connect_to_db()
            cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN ProductLine TO Product_Category;')
            cur.execute("DROP VIEW pr.p;")
            cur.execute(f'ALTER TABLE {self.new_table} ALTER COLUMN Product_Category TYPE VARCHAR(20);')

            update_query = f"""
            UPDATE {self.new_table}
            SET Product_Category = CASE
                WHEN Product_Category = 'S' THEN 'Standard'
                WHEN Product_Category = 'T' THEN 'Touring'
                WHEN Product_Category = 'M' THEN 'Mountain'
                WHEN Product_Category = 'R' THEN 'Road'
                ELSE Product_Category
            END;
            """
            cur.execute(update_query)

            cur.execute(f'ALTER TABLE {self.new_table} ALTER COLUMN Class TYPE VARCHAR(50);')
            
            update_query = f"""
            UPDATE {self.new_table}
            SET Class = CASE
                WHEN Class = 'H' THEN 'High - Premium quality and price'
                WHEN Class = 'M' THEN 'Medium - Standard quality and price'
                WHEN Class = 'L' THEN 'Low - Economy quality and price'
                ELSE Class 
            END;
            """
            cur.execute(update_query)

            cur.execute(f'ALTER TABLE {self.new_table} ALTER COLUMN Style TYPE VARCHAR(20);')
            
            update_query = f"""
            UPDATE {self.new_table}
            SET Style = CASE
                WHEN Style = 'M' THEN 'Men''s Style'
                WHEN Style = 'U' THEN 'Unisex Style'
                WHEN Style = 'W' THEN 'Women''s Style'
                ELSE Style 
            END;
            """
            cur.execute(update_query)

            conn.commit()
            conn.close()
            
            print('Categorical columns are updated successfully')
        except Exception as e: 
            print(f'Failed to update the categorical columns: {e}')

    def update_foreignkey_columns(self):
        try:  
            conn, cur = self._connect_to_db()
            cur.execute(f'ALTER TABLE {self.new_table} ADD COLUMN Product_Subcategory_Name VARCHAR(255);')
            
            update_query = f"""
            UPDATE {self.new_table} p
            SET Product_Subcategory_Name = sc.Name
            FROM production.ProductSubcategory sc
            WHERE p.ProductSubcategoryID = sc.ProductSubcategoryID;
            """
            cur.execute(update_query)

            cur.execute(f'ALTER TABLE {self.new_table} ADD COLUMN Product_Model_Name VARCHAR(255);')
            
            update_query = f"""
            UPDATE {self.new_table} p
            SET Product_Model_Name = pm.Name
            FROM production.ProductModel pm
            WHERE p.ProductModelID = pm.ProductModelID;
            """
            cur.execute(update_query)

            conn.commit()
            conn.close()

            print('Foreign key columns are updated successfully')
        except Exception as e:
            print(f'Failed to update the foreign key columns: {e}')
    
    def update_catalog_description(self):
        try:
            conn, cur = self._connect_to_db()
            cur.execute(f'ALTER TABLE {self.new_table} ADD COLUMN Product_Catalog_Description JSON;')

            fetch_query = f"""
            SELECT ProductModelID, CatalogDescription
            FROM production.ProductModel;
            """
            cur.execute(fetch_query)
            rows = cur.fetchall()

            for row in rows:
                pm_id, xml = row
                if xml: 
                    json = self.parse_xml_to_json(xml)
                    update_query = f"""
                    UPDATE {self.new_table}
                    SET Product_Catalog_Description = %s
                    WHERE ProductModelID = %s;    
                    """
                    cur.execute(update_query, (json, pm_id))

            conn.commit()
            print('Product catalog description is successfully added')
        except Exception as e:
            print(f'Failed to update the product catalog description: {e}')

    def parse_xml_to_json(self, xml_data):
        root = ET.fromstring(xml_data)

        def process_element(element):
            tag = element.tag.split('}')[-1]  # Extract tag without namespace
            element_data = {}

            # Process attributes
            for attr, value in element.attrib.items():
                attr_tag = attr.split('}')[-1]  # Extract attribute tag without namespace
                element_data[attr_tag] = value

            # Process children
            children = list(element)
            if children:
                for child in children:
                    child_tag, child_content = process_element(child)
                    if isinstance(child_content, dict) and 'text' in child_content and len(child_content) == 1:
                        # If the child only contains text, use the text directly
                        child_content = child_content['text']
                    element_data[child_tag] = child_content
            else:
                # If there are no children, store the text content
                text = element.text.strip() if element.text and element.text.strip() else None
                if text:
                    element_data['text'] = text

            return tag, element_data

        _, processed_data = process_element(root)
        return json.dumps(processed_data)

    def drop_columns(self):
        conn, cur = self._connect_to_db()

        columns_to_drop = ['productnumber', 'productsubcategoryid', 'rowguid', 'productmodelid']
        drop_query = f"ALTER TABLE {self.new_table} " + ", ".join([f"DROP COLUMN {column}" for column in columns_to_drop]) + ";"

        try: 
            cur.execute(drop_query)
            conn.commit()

            print('Successfully dropped columns')
        except Exception as e:
            print(f'failed to delete columns: {e}')
    
    def update_weight_unit_measure_code(self):
        conn, cur = self._connect_to_db()

        cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN SizeUnitMeasureCode TO Size_Unit_Measure_Code;')

        cur.execute(f'ALTER TABLE {self.new_table} RENAME COLUMN WeightUnitMeasureCode TO Weight_Unit_Measure_Code;')
        cur.execute(f'ALTER TABLE {self.new_table} ALTER COLUMN Weight_Unit_Measure_Code TYPE VARCHAR(20);')
        update_query = f"""
        UPDATE {self.new_table}
        SET Weight_Unit_Measure_Code = 'Gram'
        WHERE Weight_Unit_Measure_Code = 'G';    
        """
    
        try: 
            #cur.execute(update_query)
            conn.commit()

            print('Successfully updated the weightunitmeasurecode column')
        except Exception as e:
            print(f'failed to update the weightunitmeasurecode column: {e}')
