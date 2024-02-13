import numpy as np
import pandas as pd
import pickle
import psycopg2
from sentence_transformers import SentenceTransformer 


class DuplicateFinder:
    
    DB_NAME = 'Adventureworks'
    DB_USER = 'postgres'
    DB_PASSWORD = 'postgres'

    index_file = 'index.faiss'
    row_id_mapping_file = 'row_id_mapping.pkl'

    def __init__(self, model_name, table_name):
        import faiss # imported here because somehow it makes loading the sentence transformer crash
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.table_name = table_name
        self.row_id_mapping = []
        self.size_prompt = """The size is the most important feature to consider. 
        For example, the size 42 is very different from 48."""
        self.subcategory_prompt = """The product subcategory is the most important feature to consider.
        For example, bottom brackets are different from brakes.
        After the product subcategory, the size is important. For example, the size 42 is very different from 48. 
        """
    
    def _connect_to_db(self):
        try: 
            conn = psycopg2.connect(
                dbname=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASSWORD,
                host='localhost',
                port='5432'
            )
            cur = conn.cursor()
            return conn, cur 
        except Exception as e:
            print(f'Failed to connect to the db: {e}')
            return 

    def _fetch_rows(self):
        conn, cur = self._connect_to_db()
        cur.execute(f'SELECT * FROM {self.table_name};')

        column_names = [desc[0] for desc in cur.description]
        rows = [dict(zip(column_names, row)) for row in cur.fetchall()]
        
        conn.close()
        return rows
    
    def _add_duplicate_row(self, row):
        duplicate = []
        for key, value in row.items():
            value = str(value)
            if key == 'name':
                value = value.lower()

            duplicate.append(f'{key}: {value}')
 
        dup_text = ' '.join(duplicate)
        embedding = self.model.encode(dup_text)
        np_embedding = embedding.astype('float32').reshape(1, -1)

        self.index.add(np_embedding)
        self.row_id_mapping.append('test')

        print('The duplicate row is successfully added')

    def generate_embeddings(self):
        """
        Generates embeddings for each row and index it to faiss for similarity search. 
        An index file and a pickle file of the row id mapping are created as a result. 
        """
        import faiss 
        
        rows = self._fetch_rows()
    
        for row in rows:
            text = []
            for key, value in row.items():
                value = str(value)
                """
                if key == 'size':
                    text.extend([f'{key}: {value})']*2)
                else:
                """
                text.append(f'{key}: {value}') 

            combined_text = ' '.join(text)
            embedding = self.model.encode(combined_text)
            np_embedding = embedding.astype('float32').reshape(1, -1)
            
            self.index.add(np_embedding)
            self.row_id_mapping.append(row['productid'])
            
            if row['productid'] == 994: # add the duplicate row
                self._add_duplicate_row(row)

        faiss.write_index(self.index, self.index_file)

        with open(self.row_id_mapping_file, 'wb') as f:
            pickle.dump(self.row_id_mapping, f)
        
        print('Embeddings are generated successfully.')
        
    def _create_similarity_matrix(self):
        import faiss 
        
        index = faiss.read_index(self.index_file)
        
        with open(self.row_id_mapping_file, 'rb') as f:
            row_id_mapping = pickle.load(f)

        num_rows = index.ntotal
        similarity_matrix = np.zeros((num_rows, num_rows))
        for i in range(num_rows):
            for j in range(i+1, num_rows):
                embedding_i = index.reconstruct(i).reshape(1, -1)
                embedding_j = index.reconstruct(j).reshape(1, -1)
                
                # Calculate distance between embedding_i and embedding_j directly
                # FAISS IndexFlatL2 does not provide a method to directly calculate the distance between two vectors
                # So, we use numpy to calculate the L2 distance
                dist = np.linalg.norm(embedding_i - embedding_j)
                
                # Assign the calculated distance to both [i, j] and [j, i] to ensure symmetry
                similarity_matrix[i, j] = dist
                similarity_matrix[j, i] = dist

        np.fill_diagonal(similarity_matrix, 0)
        df = pd.DataFrame(similarity_matrix, index=row_id_mapping, columns=row_id_mapping)

        print('Similarity matrix is successfully created.')
        return df

    def extract_lowest_distances(self, n=5):
        df = self._create_similarity_matrix()
        
        # Use the mask to exclude diagonal values
        mask = np.ones(df.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        flattened_df = df.values[mask]
        row_indices, col_indices = np.where(mask)

        # Filter out redundant pairs by keeping only those where row index < col_index
        filtered_indices= row_indices < col_indices
        filtered_df = flattened_df[filtered_indices]
        filtered_rows = row_indices[filtered_indices]
        filtered_cols = col_indices[filtered_indices]

        sorted_indices = np.argsort(filtered_df)
        top_n_indices = sorted_indices[:n]
        top_n_distances = flattened_df[top_n_indices]

        top_n_row_ids = df.index[filtered_rows[top_n_indices]]
        top_n_col_ids = df.columns[filtered_cols[top_n_indices]]

        top_n_pairs = [(row_id, col_id, dist) for row_id, col_id, dist in zip(top_n_row_ids, top_n_col_ids, top_n_distances)]
        for i, (row_id, col_id, dist) in enumerate(top_n_pairs, 1):
            print(f"Top {i}: {row_id} to {col_id} with distance {dist}")

    def extract_distance_between_pairs(self, pair_ids):
        df = self._create_similarity_matrix()

        print(f'Prompt: {self.size_prompt}')

        distance = df.at[pair_ids[0][0], pair_ids[0][1]]
        print(f"Distance between {pair_ids[0][0]} and {pair_ids[0][1]}: {distance}")

        distance = df.at[pair_ids[1][0], pair_ids[1][1]]
        print(f"Distance between {pair_ids[1][0]} and {pair_ids[1][1]}: {distance}")

        distance = df.at[pair_ids[2][0], pair_ids[2][1]]
        print(f"Distance between {pair_ids[2][0]} and {pair_ids[2][1]}: {distance}")


TABLE_NAME = 'production.product_flattened'
duplicate_finder = DuplicateFinder('sentence-transformers/all-MiniLM-L6-v2', TABLE_NAME)
duplicate_finder.generate_embeddings()
duplicate_finder.extract_lowest_distances()
#duplicate_finder.extract_distance_between_pairs([[965, 964], [964, 961], [965, 961]])