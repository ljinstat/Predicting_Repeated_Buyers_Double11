import os
import pandas as pd
import rampwf as rw
import pdb
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Taobao: Repeat Buyers Prediction'
_target_column_name = 'label'
#_ignore_column_names = ['user_id', 'merchant_id']
_prediction_label_names = [0, 1]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()


class BaseScoreType(object):
    def check_y_pred_dimensions(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should have {} instances, '
                'instead it has {} instances'.format(len(y_true), len(y_pred)))

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths.y_pred[valid_indexes]
        y_pred = predictions.y_pred[valid_indexes]
        self.check_y_pred_dimensions(y_true, y_pred)
        return self.__call__(y_true, y_pred)

class ClassifierBaseScoreType(BaseScoreType):
    def score_function(self, ground_truths, predictions, valid_indexes=None):
        self.label_names = ground_truths.label_names
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
        y_true_label_index = ground_truths.y_pred_label_index[valid_indexes]
        self.check_y_pred_dimensions(y_true_label_index, y_pred_label_index)
        return self.__call__(y_true_label_index, y_pred_label_index)

class businessmetric(ClassifierBaseScoreType):
    
    def __init__(self, name='business_metric', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(len(y_pred)): 
            if y_true[i]==y_pred[i]==1:
                TP += 1
                
        for i in range(len(y_pred)): 
            if y_pred[i]==1 and y_true[i]!=y_pred[i]:
                FP += 1
                
        for i in range(len(y_pred)): 
            if y_true[i]==y_pred[i]==0:
                TN += 1
                
        for i in range(len(y_pred)): 
            if y_pred[i]==0 and y_true[i]!=y_pred[i]:
                FN += 1
        
        promotion_per_client = 1.
        proba = 0.3
        margin_per_client = 5.

        cfn = FN * margin_per_client - FN * promotion_per_client
        cfp = - FP * proba * margin_per_client
        ctn = TN * proba * margin_per_client - TN * promotion_per_client
        ctp = TP * margin_per_client
        
        return (cfp*FP + cfn*FN + ctp*TP + ctn*TN)/(TP+TN+FP+FN)

        

score_types = [   businessmetric(),
    rw.score_types.Accuracy(name='acc', precision=3),
    rw.score_types.F1Above(name='f1_70', threshold=0.7),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
    rw.score_types.ROCAUC(name='roc_auc', precision=3)
    ]



def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name], axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


