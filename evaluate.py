from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from inspect import signature
import matplotlib.pyplot as plt
from features import ESGData
import torch
import torch.nn as nn
import pandas as pd
import mlflow
import json
import time
from models import ModelConfig, ESGClassifier

class PredictModel:

    def __init__(self, config, model, feat):
        self.config = config
        self.model = model
        self.feat = feat

        self._initialize_device()

    def _initialize_device(self):

        if torch.cuda.is_available():
            device = torch.device('cuda')
            print('There are {} GPU(s) available.'.format(torch.cuda.device_count()))
            print('We will use the GPU: {}'.format(torch.cuda.get_device_name(0)))
        else:
            print('No GPU Available, using the CPU instead.')
            device = torch.device('cpu')

        self.device = device
        self.model.to(device)

    def generate_predictions(self):

        self.model.eval()

        with torch.no_grad():

            predictions = []
            dataloader = self.feat.test_data_loader

            for batch in dataloader:

                input_ids = batch[0].long().to(self.device)
                outputs = self.model(input_ids)

                predictions.append(outputs[1].detach().cpu().numpy())

        flat_predictions = [item for sublist in predictions for item in sublist]
        self.predictions = pd.DataFrame(flat_predictions, columns=self.feat.label_cols)

        # Save Predictions Locally
        self.predictions.to_csv(self.config.pred_filename, index=False)

        if self.config.mlflow_tracking: mlflow.log_artifact(self.config.pred_filename)

class EvaluateModel:

    def __init__(self, config, feat):
        self.config = config
        self.feat = feat

        # Always evaluate test results
        self.test_results = {}
        for label in self.feat.label_cols:
            self.test_results[label] = {}

        # Evaluate against train results if evaluate_train_output=True
        if self.config.evaluate_train_output:
            self.train_results = {}
            for label in self.feat.label_cols:
                self.train_results[label] = {}


    def _calculate_pr(self, evaluate_df, predictions):

        pr_dictionary = {}
        for label in self.feat.label_cols:
            pr_dictionary[label] = {}

        # Iterate through each label
        for label in self.feat.label_cols:

            # Calculate precision, recall
            precision, recall, thresholds = precision_recall_curve(evaluate_df[label],
                                                                   predictions[label])

            # Calculate PR AUC
            pr_auc = auc(recall, precision)

            # Add to Result Dictionary
            pr_dictionary[label]['precision'] = precision
            pr_dictionary[label]['recall'] = recall
            pr_dictionary[label]['pr_auc'] = pr_auc

        return pr_dictionary

    def _calculate_roc(self, evaluate_df, predictions):

        roc_dictionary = {}
        for label in self.feat.label_cols:
            roc_dictionary[label] = {}

        # Iterate through each label
        for label in self.feat.label_cols:

            # Calculate fpr/tpr
            fpr, tpr, thresholds = roc_curve(evaluate_df[label],
                                             predictions[label])

            # Calculate ROC AUC
            roc_auc = roc_auc_score(evaluate_df[label],
                                    predictions[label])

            # Add to Result Dictionary
            roc_dictionary[label]['fpr'] = fpr
            roc_dictionary[label]['tpr'] = tpr
            roc_dictionary[label]['roc_auc'] = roc_auc

        return roc_dictionary


    def plot_auc(self, x_axis, y_axis, auc, label, x_axis_label, y_axis_label, curve_name, train_or_test):

        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})

        plt.figure(figsize=(8, 6))
        plt.style.use('seaborn-paper')
        plt.step(x_axis, y_axis, color='b', alpha=0.5, where='post')

        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(label + ' ' + curve_name + ': AUC{0:0.2f}'.format(auc))
        filename = r'esg_model_data/figures/' + train_or_test + '_' + curve_name + '_' + label + '.pdf'

        plt.savefig(filename, bbox_inches='tight')
        time.sleep(5)

        plt.close()

        if self.config.mlflow_tracking: mlflow.log_artifact(filename)

    def calculate_all_metrics(self):

        # Get Test Metrics
        test_evaluate_df = self.feat.test_df
        test_predictions = pd.read_csv(self.config.pred_filename)
        test_pr_dictionary = self._calculate_pr(test_evaluate_df, test_predictions)
        test_roc_dictionary = self._calculate_roc(test_evaluate_df, test_predictions)

        for label in self.feat.label_cols:
            for key in test_pr_dictionary[label].keys():
                self.test_results[label][key] = test_pr_dictionary[label][key]

            for key in test_roc_dictionary[label].keys():
                self.test_results[label][key] = test_roc_dictionary[label][key]

        if self.config.evaluate_train_output:

            train_evaluate_df = self.feat.train_df
            train_predictions = pd.read_csv(self.config.output_filename)
            train_pr_dictionary = self._calculate_pr(train_evaluate_df, train_predictions)
            train_roc_dictionary = self._calculate_roc(train_evaluate_df, train_predictions)

            for label in self.feat.label_cols:
                for key in train_pr_dictionary[label].keys():
                    self.train_results[label][key] = train_pr_dictionary[label][key]
                for key in train_roc_dictionary[label].keys():
                    self.train_results[label][key] = train_roc_dictionary[label][key]

    def generate_all_figures(self):

        # Plot all PR Curves
        for label in self.feat.label_cols:
            x_axis = self.test_results[label]['recall']
            y_axis = self.test_results[label]['precision']
            auc = self.test_results[label]['pr_auc']
            self.plot_auc(x_axis, y_axis, auc, label, 'Recall', 'Precision', 'Precision-Recall curve',
                          train_or_test='test')

        # Plot all ROC Curves
        for label in self.feat.label_cols:
            x_axis = self.test_results[label]['fpr']
            y_axis = self.test_results[label]['tpr']
            auc = self.test_results[label]['roc_auc']
            self.plot_auc(x_axis, y_axis, auc, label, 'TPR', 'FPR', 'ROC Curve', train_or_test='test')

        if self.config.evaluate_train_output:

            # Plot all Train PR Curves
            for label in self.feat.label_cols:
                x_axis = self.train_results[label]['recall']
                y_axis = self.train_results[label]['precision']
                auc = self.train_results[label]['pr_auc']
                self.plot_auc(x_axis, y_axis, auc, label, 'Recall', 'Precision', 'Precision-Recall curve',
                              train_or_test='train')

            # Plot all Train ROC Curves
            for label in self.feat.label_cols:
                x_axis = self.train_results[label]['fpr']
                y_axis = self.train_results[label]['tpr']
                auc = self.train_results[label]['roc_auc']
                self.plot_auc(x_axis, y_axis, auc, label, 'TPR', 'FPR', 'ROC Curve', train_or_test='test')

if __name__ == '__main__':

    layer_dict = {1: {'input_size': 768,
                      'output_size': 168,
                      'dropout': 0.25,
                      'activation_fn': nn.ReLU()}}

    config = ModelConfig(model_type='BERT',
                         pretrained_base='bert-base-uncased',
                         mlflow_tracking=False,
                         epochs=1,
                         layer_dict=layer_dict,
                         max_seq_length=64,
                         batch_size=12,
                         pretrained_dropout=0.1,
                         num_labels=10,
                         model_filename='results/model.pt',
                         pred_filename='results/predictions.pt',
                         evaluate_every=100)

    model = ESGClassifier(config)

    feat = ESGData(config)
    feat.execute()

    pm = PredictModel(config, model, feat)
    pm.generate_predictions()

    em = EvaluateModel(config, feat)
    em.calculate_all_metrics()
    em.generate_all_figures()

