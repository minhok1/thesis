from models import ESGClassifier, ModelConfig
from features import ESGData
from train import TrainModel
from evaluate import PredictModel, EvaluateModel
import mlflow
import torch.nn as nn

def initialize_mlflow(config):

    mlflow.set_experiment(config.model_type)
    mlflow.start_run()

    mlflow.log_param('epochs', config.epochs)

    if len(config.layer_dict.keys()) > 0:
        mlflow.log_param('hidden_layers', list(config.layer_dict.keys())[-1])
        mlflow.log_param('layer_size', config.layer_dict[1]['output_size'])
    else:
        mlflow.log_param('hidden_layers', 0)


    mlflow.log_param('max_seq_length', config.max_seq_length)
    mlflow.log_param('pretrained_base', config.pretrained_base)
    mlflow.log_param('batch_size', config.batch_size)

    runid = mlflow.active_run().info.run_id

    print('MLFlow Tracking Enabled: {}'.format(runid))

    config.model_filename = 'esg_model_data/results/model_' + str(runid) + '.pt'
    config.pred_filename = 'esg_model_data/results/pred_' + str(runid) + '.csv'
    config.output_filename = 'esg_model_data/results/outputs_' + str(runid) + '.csv'

    return config

def run_experiment(config):

    # Initialize MLFlow and update filenames if tracking
    if config.mlflow_tracking:
        config = initialize_mlflow(config)

    # Initialize Model
    model = ESGClassifier(config)

    # Initialize Features
    feat = ESGData(config)
    feat.execute()

    # Run Training
    tm = TrainModel(config, model, feat)
    tm.train()

    # Generate Test Predictions
    pm = PredictModel(config, tm.best_model, feat)
    pm.generate_predictions()

    # Evaluate Model
    em = EvaluateModel(config, feat)
    em.calculate_all_metrics()
    em.generate_all_figures()

if __name__ == '__main__':

    # layer_dict = {1: {'input_size': 768,
    #                   'output_size': 256,
    #                   'dropout': 0.25,
    #                   'activation_fn': nn.ReLU()}}

    layer_dict = {}

    config = ModelConfig(model_type='BERT',
                         pretrained_base='bert-base-uncased',
                         mlflow_tracking=True,
                         epochs=10,
                         layer_dict=layer_dict,
                         max_seq_length=256,
                         batch_size=16,
                         pretrained_dropout=0,
                         num_labels=27,
                         model_filename='esg_model_data/results/model_bert.pt',
                         pred_filename='esg_model_data/results/pred_bert.csv',
                         output_filename='esg_model_data/results/outputs_bert.csv',
                         evaluate_every=400,
                         consecutive_evals_no_improvement_to_stop=5,
                         )

    run_experiment(config)