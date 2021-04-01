# Import packages
from transformers import AlbertModel, AlbertPreTrainedModel, AlbertConfig, BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
#from features import ESGData

class ModelConfig:
    def __init__(self, model_type, pretrained_base, mlflow_tracking=False, epochs=5, layer_dict=None, max_seq_length=64,
                 batch_size=12, pretrained_dropout=0.1, num_labels=10, model_filename=None, pred_filename=None,
                 output_filename=None, evaluate_every=100, evaluate_train_output=False,
                 consecutive_evals_no_improvement_to_stop=3):
        self.model_type = model_type
        self.pretrained_base = pretrained_base
        self.epochs = epochs
        self.mlflow_tracking = mlflow_tracking
        self.layer_dict = layer_dict
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.pretrained_dropout = pretrained_dropout
        self.num_labels = num_labels
        self.model_filename = model_filename
        self.pred_filename = pred_filename
        self.output_filename = output_filename
        self.evaluate_every = evaluate_every
        self.evaluate_train_output = evaluate_train_output
        self.consecutive_evals_no_improvement_to_stop = consecutive_evals_no_improvement_to_stop

        assert(self.model_type in ['BERT','ALBERT'])

class ActivatedLinearWithDropout(nn.Module):
    def __init__(self, input_size, output_size, dropout, activate_fn):
        super(ActivatedLinearWithDropout, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = activate_fn

    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        return outputs

class ESGClassifier(nn.Module):
    def __init__(self, config):
        super(ESGClassifier, self).__init__()
        self.config = config

        if self.config.model_type == 'ALBERT':
            self.pretrained_model = AlbertModel.from_pretrained(self.config.pretrained_base)
            hidden_state_size = 768
        if self.config.model_type == 'BERT':
            self.pretrained_model = BertModel.from_pretrained(self.config.pretrained_base)
            hidden_state_size = 768

        self.pretrained_dropout = nn.Dropout(self.config.pretrained_dropout)
        self.pretrained_model.init_weights()

        #Initialise Hidden Layers
        final_output_size = None
        for layer in self.config.layer_dict.keys():
            fc = ActivatedLinearWithDropout(self.config.layer_dict[layer]['input_size'],
                                            self.config.layer_dict[layer]['output_size'],
                                            self.config.layer_dict[layer]['dropout'],
                                            self.config.layer_dict[layer]['activation_fn'])
            setattr(self, 'fc_' + str(layer), fc)
            final_output_size = self.config.layer_dict[layer]['output_size']

            # Initialize Final Output layer
        if final_output_size is None:
            final_output_size = hidden_state_size

        self.output_layer = nn.Linear(final_output_size, self.config.num_labels)

        # Define Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, b_input_ids, token_type_ids=None, attention_mask=None):

        # Pass through base albert
        output = self.pretrained_model(b_input_ids, token_type_ids, attention_mask)
        logits = output[1]
        logits = self.pretrained_dropout(logits)

        # Pass through hidden layers
        for layer in self.config.layer_dict.keys():
            fc_layer = getattr(self, 'fc_' + str(layer))
            logits = fc_layer(logits)

        # Pass through final connected layer
        logits = self.output_layer(logits)
        sigmoids = self.sigmoid(logits)

        return logits, sigmoids

# if __name__ == '__main__':
#     # Initialize Config
#     config = ModelConfig(model_type='BERT',
#                          pretrained_base='bert-base-uncased',
#                          layer_dict={1: {'input_size': 768,
#                                          'output_size': 168,
#                                          'dropout': 0.25,
#                                          'activation_fn': nn.ReLU()}},
#                          max_seq_length=64,
#                          batch_size=16,
#                          pretrained_dropout=0.1,
#                          num_labels=10)
#
#     # Initialize Features
#     feat = ESGData(config)
#     feat.execute()
#
#     # Initialize Model and Test Forward Loop
#     model = ESGClassifier(config)
#
#     # Test Forward Loop
#     model.eval()
#     for step, batch in enumerate(feat.train_data_loader):
#         batch_inputs, batch_token_types, batch_attention_masks = batch
#         with torch.no_grad():
#             logits, sigmoids = model(b_input_ids=batch_inputs, token_type_ids=None, attention_mask=None)
#             break
#         break
#
#     print(logits)
#     print(sigmoids)