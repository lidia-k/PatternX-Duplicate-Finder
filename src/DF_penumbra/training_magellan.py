import joblib
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py_entitymatching as em


class MagellanTrainer:

    matchers = {
        'dt': em.DTMatcher(name='DecisionTree', random_state=0),
        'svm': em.SVMMatcher(name='SVM', random_state=0),
        'rf': em.RFMatcher(name='RF', random_state=0),
        'lg': em.LogRegMatcher(name='LogReg', random_state=0),
        'ln': em.LinRegMatcher(name='LinReg'),  
        'nb': em.NBMatcher(name='NaiveBayes')
    }
    attrs_after = None
    exclude_attrs = ['id', 'ltable_id', 'rtable_id']

    def __init__(self, ltable=None, rtable=None, data=None, model=None, training=False):
        self.ltable = ltable
        self.rtable = rtable

        self.data = data
        if training: 
            self.train_set, self.test_set = self._split_data()

        self._load_model(model)
        self.feature_table = self._load_feature_table(training)
    
    def _load_model(self, model):
        if type(model) == str:
            self.model = self.matchers[model]
        else:  
            self.model = model  

    def _load_feature_table(self, training):
        if training:
            feature_table = em.get_features_for_matching(
                self.ltable, self.rtable, validate_inferred_attr_types=False
            )
            with open('feature_table.pkl', 'wb') as f:
                dill.dump(feature_table,  f)
            return feature_table
        
        try: 
            with open('feature_table.pkl', 'rb') as f:
                feature_table = dill.load(f)
            return feature_table
        except Exception as e:
            print('Feature table not found')
            sys.exist(1)

    def _split_data(self):
        split_data = em.split_train_test(self.data, train_proportion=0.7, random_state=0)
        train_set = split_data['train']
        test_set = split_data['test']
        return train_set, test_set
    
    def _create_features(self, dataset):
        f_vectors = em.extract_feature_vecs(
            dataset, feature_table=self.feature_table, attrs_after=self.attrs_after, show_progress=True
        )        
        if any(pd.notnull(f_vectors)):
            f_vectors = em.impute_table(
                f_vectors, 
                exclude_attrs=self.exclude_attrs, 
                strategy='mean'
            )
        
        return f_vectors

    def _select_best_model(self, f_vectors):
        result = em.select_matcher(
            [matcher for matcher in self.matchers.values()], 
            table=f_vectors, 
            exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
            k=5, target_attr=self.attrs_after, 
            metric_to_select_matcher='f1', 
            random_state=0)
        
        print(result['cv_stats'])
        return result['selected_matcher']
    
    def debug_model(self):
        em.vis_debug_rf(self.models[2], self.train_set, self.test_set, 
                        exclude_attrs=self.exclude_attrs,
                        target_attr='label')

    def train_model(self):
        self.attrs_after = 'label'
        self.exclude_attrs.append(self.attrs_after)
        f_vectors = self._create_features(self.train_set)
        
        best_model = self._select_best_model(f_vectors)
        # You can choose to use the best model or a specific model. We're currently using a specific model.
        self.model.fit(
            table=f_vectors, 
            exclude_attrs=self.exclude_attrs, 
            target_attr='label'
        )
        joblib.dump(self.model, 'model.pkl')

    def predict(self, data=None):
        if data is None:
            data = self.test_set

        f_vectors = self._create_features(data)
        predictions = self.model.predict(
            table=f_vectors, 
            exclude_attrs=self.exclude_attrs, 
            append=True, target_attr='predicted', inplace=False
        )

        # Save predictions to a CSV file
        merge_df = data.merge(predictions[['id', 'predicted']], on='id', how='left')
        merge_df = merge_df[['id', 'predicted', 'ltable_fullname', 'rtable_fullname', 
                             'ltable_email', 'rtable_email', 'ltable_sap_no', 'rtable_sap_no']]
        merge_df.to_csv(f'predictions_{self.model.clf.__class__.__name__}.csv', index=False)
        return predictions
     
    def evaluate(self, predictions):
        eval_result = em.eval_matches(predictions, 'label', 'predicted')
        em.print_eval_summary(eval_result)
    
    def retrieve_feature_importance(self):
        importances = self.model.clf.feature_importances_
        feature_names = self.feature_table['feature_name'].values
        
        plt.figure(figsize=(10, 15))
        indices = np.argsort(importances)[::-1][:30]

        plt.title(f'Feature Importance in {self.model.clf.__class__.__name__}')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')

        plt.subplots_adjust(left=0.3)
        plt.show()
        