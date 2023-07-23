import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, classification_report, confusion_matrix, log_loss
from flask import Flask, jsonify, request
import time
from time import perf_counter
from codecarbon import track_emissions


class ClassificationModelHelper():
    
    def __init__(self, experiment_name, model_dict = None):
        # set experiment name for mlflow
        self._experiment_name = experiment_name
        
        if not model_dict == None:
            # model dictionary containing estimator object and grid_search parameters
            self._model_dict = model_dict
            return None
        
        # this will only run if model_dict is None
        rfc = RandomForestClassifier()
        gbc = GradientBoostingClassifier()
        xgbc = XGBClassifier()
        
        rfc_params = {'n_estimators':[250, 500, 750, 1000], 'max_depth':[3,5,7,9], 'min_samples_split':[2,4,6,8], 'bootstrap':[True,False]}      
        gbc_params = {'learning_rate':[0.1, 0.01, 0.001, 0.0001, 0.00001], 'n_estimators':[250, 500, 750, 1000], 'max_depth':[3,5,7,9], 'min_samples_split':[2,4,6,8]}
        xgbc_params = {'learning_rate':[0.1, 0.01, 0.001, 0.0001, 0.00001], 'n_estimators':[250, 500, 750, 1000], 'max_depth':[3,5,7,9]}
        
        # correct format of model_dict
        self._model_dict = {rfc: [rfc_params, 'RandomForest Classifier'], gbc: [gbc_params, 'GradientBoosting Classifier'], xgbc: [xgbc_params, 'XGBoost Classifier']}
        
        return None
    
    def show_model_summary(self, model_name, y_true, y_pred, y_pred_prob):
        # Get model metrics after training
        print('\n\n===========================================\n\n')
        
        print('%s model summary' %model_name)
        print('Classification Report: ')
        print(classification_report(y_true, y_pred))
        print('Confusion Matrix: ')
        print(confusion_matrix(y_true, y_pred))
        print('Logloss: ', log_loss(y_true, y_pred_prob[:, 1]))
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)        
        auc_score = roc_auc_score(y_true, y_pred_prob[:, 1])
        print('auc: ', auc_score)
        
        print('\n\n===========================================\n\n')
        # return metrics to log metrics into mlflow
        model_metrics = {'accuracy': accuracy, 'precision': precision, 'f1_score': f1, 'log_loss': log_loss(y_true, y_pred_prob[:, 1]), 'auc': auc_score}
        return model_metrics
        

    def _run(self, model, X_train, X_test, y_train, y_test, cross_validation, verbose):
        # get model object and parameters
        model_obj = model
        model_params = self._model_dict[model][0]
        model_name = self._model_dict[model][1]
        
        # hyper-parameter tuning with cross-validation
        grid_model = GridSearchCV(model_obj, param_grid=model_params, cv=cross_validation, verbose=verbose, n_jobs=2)
        print('Training %s \n' %model_name)
        grid_model.fit(X_train, y_train)
        print('Training finished.\n\n')
            
        print('%s Best Params: ' %model_name)
        model_params = grid_model.best_params_
        print(model_params)
            
        y_pred = grid_model.predict(X_test)
        y_pred_prob = grid_model.predict_proba(X_test)
        model_metrics = self.show_model_summary(model_name, y_test, y_pred, y_pred_prob)
             
        model_experiment_summary = {'run_name': model_name, 'run_params': model_params, 'run_metrics': model_metrics}
        return model_experiment_summary
                
    
    def train_model(self, X_train, X_test, y_train, y_test, n_folds=5, verbose=1):
        # entry point method
        start_time = perf_counter()
        print('Starting experiment: %s' %self._experiment_name)
        print('\n')
        model_experiment_summaries = []
        for model in self._model_dict.keys():
            # call run method to train model - return metrics dictionary
            model_experiment_summary = self._run(model, X_train, X_test, y_train, y_test, n_folds, verbose)
            model_experiment_summaries.append(model_experiment_summary)
            print('\n\n')
        end_time = perf_counter()
        minutes_taken = round((float(end_time-start_time)/60.0), 2)
        print('\nTotal time taken = %.2f minutes' %minutes_taken) 
        return model_experiment_summaries 


def run_train_pipeline(experiment_name, dia_df, label):
    '''
    method which executes the training pipeline
    params:
        dia_df : diabetes dataframe
        tol : C (tolerance) value for regularization
        iterations : maximum iterations for training Logistic Regression model
        experiment_name : MLflow experiment name
        run_name : Set run name inside each experiment
    '''
    X_train, X_test, y_train, y_test = train_test_split(dia_df.drop(label, axis=1), dia_df[label], test_size=0.3, random_state=123)
    model = ClassificationModelHelper(experiment_name=experiment_name)
    return model.train_model(X_train, X_test, y_train, y_test)


app = Flask(__name__)

@app.route('/train', methods=['POST'])
@track_emissions(offline=True, project_name="diabetes_ml_train", country_iso_code="USA", region="California")
def train():
    if request.method == 'POST':
        label = request.form['label']
        csv_file = request.files['train_file']
        if csv_file == None:
            raise ValueError('Train File is not provided.')
        file_name = csv_file.filename
        if file_name.split('.')[1] == 'csv':
            clock_time = time.ctime().replace(' ', '-')
            save_file = './temp/' + file_name.split('.')[0] + '_' + clock_time + '.csv'
            csv_file.save(save_file)
            experiment_name = request.form['experiment_name']
            dia_df = pd.read_csv(save_file)
            metrics = run_train_pipeline(experiment_name, dia_df, label)
            return jsonify(metrics)
    else:
        raise TypeError('Invalid file type. App only accepts csv files.')

'''
data = request.get_json()
df = pd.DataFrame(data, index=[0])
'''


if __name__ == '__main__':
    # create temp folder
    if not os.path.exists('./temp/'):
        os.makedirs('./temp/')
    # start flask app
    app.run(debug=True, port=5005)