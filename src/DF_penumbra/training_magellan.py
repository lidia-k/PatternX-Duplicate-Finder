import joblib
import sys

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py_entitymatching as em
from sklearn.model_selection import GridSearchCV

class MagellanTrainer:

    matchers = {
        'dt': em.DTMatcher(name='DecisionTree', random_state=0),
        'svm': em.SVMMatcher(name='SVM', random_state=0),
        'rf': em.RFMatcher(name='RF', random_state=0, **{'max_depth': 10, 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 50}),
        'lg': em.LogRegMatcher(name='LogReg', random_state=0),
        'ln': em.LinRegMatcher(name='LinReg'),  
        'nb': em.NBMatcher(name='NaiveBayes'),
        'xgb': em.XGBoostMatcher(name='XGBoost', random_state=0)
    }
    attrs_after = None
    exclude_attrs = ['id', 'ltable_id', 'rtable_id']

    def __init__(self, ltable=None, rtable=None, data=None, model=None, training=False):
        self.ltable = ltable
        self.rtable = rtable

        self.data = data
        if training: 
            self.train_set, self.test_set = self._split_data()
        self.model_name = model
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
        self.attrs_after = 'label'
        self.exclude_attrs.append(self.attrs_after)
        
        train_vectors = self._create_features(self.train_set)
        test_vectors = self._create_features(self.test_set)

        em.vis_debug_rf(self.model, train_vectors, test_vectors, 
                        exclude_attrs=self.exclude_attrs,
                        target_attr='label')
        
    def _perform_grid_search(self, train_vectors):
        if self.model_name == 'rf':
            param_grid = {
                'max_depth': [1, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [5, 10],
                'n_estimators': [50, 100],
            }
        elif self.model_name == 'xgb':
            param_grid = {
                'learning_rate': [0.05, 0.1, 0.3],
                'n_estimators': [50, 100],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'min_child_weight': [1, 5],
                'scale_pos_weight': [1, 2],
                #'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
                #'reg_alpha': [0, 0.01, 0.1, 1, 10],
                #'reg_lambda': [0, 0.01, 0.1, 1, 10]
            }

        x_train = train_vectors.drop(columns=self.exclude_attrs, axis=1)
        y_train = train_vectors['label']

        #cv_scores = cross_val_score(self.model.clf, x_train, y_train, cv=3, scoring='f1')

        grid_search = GridSearchCV(
            estimator=self.model.clf, 
            param_grid=param_grid, 
            scoring='f1',
            cv=3,
            verbose=1
        )
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        return best_params        

    def train_model(self):
        self.attrs_after = 'label'
        self.exclude_attrs.append(self.attrs_after)
        f_vectors = self._create_features(self.train_set)

        # You can choose to use the best model or a specific model. We're currently using a specific model.
        # best_model = self._select_best_model(f_vectors)
        params = self._perform_grid_search(f_vectors)
        if self.model_name == 'rf':
            self.model = em.RFMatcher(
                name='RF', 
                random_state=0, 
                **params
            )
        if self.model_name == 'xgb':
            self.model = em.XGBoostMatcher(
                name='XGBoost', 
                random_state=0, 
                **params
            )

        print(f'Training {self.model_name} model with {self.model.clf.get_params()}')
        self.model.fit(
            table=f_vectors, 
            exclude_attrs=self.exclude_attrs, 
            target_attr='label'
        )
        joblib.dump(self.model, 'model.pkl')

    def predict(self, data=None, all=False):
        if data is None:
            data = self.test_set

        f_vectors = self._create_features(data)
        predictions = self.model.predict(
            table=f_vectors,
            exclude_attrs=self.exclude_attrs,
            append=True,
            target_attr="predicted",
            inplace=False,
            return_probs=True, 
            probs_attr='prob'
        )

        # Save predictions to a CSV file
        merge_df = data.merge(predictions[['id', 'predicted', 'prob']], on='id', how='left')
        #merge_df = merge_df[['id', 'predicted', 'ltable_fname', 'ltable_lname',
        #                    'rtable_fname', 'rtable_lname', 'ltable_email', 'rtable_email',
        #                    'ltable_sap_no', 'rtable_sap_no']]
        filename = f'predictions_{self.model.clf.__class__.__name__}'
        if all:
            filename = filename + '_all'
        merge_df.to_csv(f'{filename}.csv', index=False)
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
        plt.xlabel('Relative Importance')

        ax = plt.gca()
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])

        for label in ax.get_yticklabels():
            if 'name' in label.get_text():
                label.set_color('red')
            else: 
                label.set_color('black')

        plt.subplots_adjust(left=0.3)
        plt.show()
