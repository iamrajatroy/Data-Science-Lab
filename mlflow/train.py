import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

import mlflow

import time
import os

from flask import Flask, jsonify, request

class TrainingPipeline(Pipeline):

    ''' 
    Class -> TrainingPipeline, ParentClass -> Sklearn-Pipeline
    Extends from Scikit-Learn Pipeline class. Additional functionality to track model metrics and log model artifacts with mlflow
    params:
        steps: list of tuple (similar to Scikit-Learn Pipeline class)
    '''
    
    def __init__(self, steps):
        super().__init__(steps)
        
    def fit(self, X_train, y_train):
        self.__pipeline = super().fit(X_train, y_train)
        return self.__pipeline
    
    def get_metrics(self, y_true, y_pred, y_pred_prob):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        entropy = log_loss(y_true, y_pred_prob)
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}

    def make_model_name(self, experiment_name, run_name):
        clock_time = time.ctime().replace(' ', '-')
        return experiment_name + '_' + run_name + '_' + clock_time
    
    def log_model(self, model_key, X_test, y_test, experiment_name, run_name, run_params=None):
        
        model = self.__pipeline.get_params()[model_key]
        
        y_pred = self.__pipeline.predict(X_test)
        y_pred_prob = self.__pipeline.predict_proba(X_test)
        
        run_metrics = self.get_metrics(y_test, y_pred, y_pred_prob)
        
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri('http://localhost:5000')
        
        with mlflow.start_run(run_name=run_name):

            if not run_params == None:
                for name in run_params:
                    mlflow.log_param(name, run_params[name])

            for name in run_metrics:
                mlflow.log_metric(name, run_metrics[name])

            model_name = self.make_model_name(experiment_name, run_name)   
            mlflow.sklearn.log_model(sk_model=self.__pipeline, artifact_path='diabetes-model', registered_model_name=model_name)
                
        print('Run - %s is logged to Experiment - %s' %(run_name, experiment_name))
        
        return run_metrics


def run_train_pipeline(dia_df, tol, iterations, experiment_name, run_name):
    '''
    method which executes the training pipeline
    params:
        dia_df : diabetes dataframe
        tol : C (tolerance) value for regularization
        iterations : maximum iterations for training Logistic Regression model
        experiment_name : MLflow experiment name
        run_name : Set run name inside each experiment
    '''
    X_train, X_test, y_train, y_test = train_test_split(dia_df.drop('Outcome', axis=1), dia_df['Outcome'], test_size=0.3, random_state=123)
    train_pipeline = TrainingPipeline(steps=[('scaler', MinMaxScaler()), ('model', LogisticRegression(C=tol, max_iter=iterations))])
    run_params = {'C': tol, 'max_iter': iterations}
    train_pipeline.fit(X_train, y_train)
    return train_pipeline.log_model('model', X_test, y_test, experiment_name, run_name, run_params=run_params)


app = Flask(__name__)

@app.route('/test', methods=['GET'])
def test_app():
    return jsonify({'response': 'Hello World'})

@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        csv_file = request.files['train_file']
        if csv_file == None:
            raise ValueError('Train File is not provided.')
        file_name = csv_file.filename
        if file_name.split('.')[1] == 'csv':
            clock_time = time.ctime().replace(' ', '-')
            save_file = './temp/' + file_name.split('.')[0] + '_' + clock_time + '.csv'
            csv_file.save(save_file)
            tol = float(request.form['tol'])
            iterations = int(request.form['iterations'])
            experiment_name = request.form['exp_name']
            run_name = 'Train-Run-' + clock_time
            dia_df = pd.read_csv(save_file)
            metrics = run_train_pipeline(dia_df, tol, iterations, experiment_name, run_name)
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