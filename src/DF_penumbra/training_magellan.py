import pandas as pd
import py_entitymatching as em


class MagellanTrainer:
    def __init__(self, ltable, rtable, trainig_data):
        self.ltable = ltable
        self.rtable = rtable
        self.training_data = trainig_data
        self.models = [
            em.DTMatcher(name='DecisionTree', random_state=0),
            em.SVMMatcher(name='SVM', random_state=0),
            em.RFMatcher(name='RF', random_state=0),
            em.LogRegMatcher(name='LogReg', random_state=0),
            em.LinRegMatcher(name='LinReg'),
        ]
    
    def train_model(self):
        split_data = em.split_train_test(self.training_data, train_proportion=0.7, random_state=0)
        train_set = split_data['train']
        test_set = split_data['test']

        f_table = em.get_features_for_matching(self.ltable, self.rtable, validate_inferred_attr_types=False)
        f_vectors = em.extract_feature_vecs(train_set, feature_table=f_table, attrs_after='label', show_progress=True)
        if any(pd.notnull(f_vectors)):
            f_vectors = em.impute_table(
                f_vectors, 
                exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
                strategy='mean'
            )

        result = em.select_matcher(
            self.models, table=f_vectors, 
            exclude_attrs=['id', 'ltable_id', 'rtable_id', 'label'], 
            k=5, target_attr='label', metric_to_select_matcher='f1', 
            random_state=0)
        print(result['cv_stats'])