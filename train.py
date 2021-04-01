import time
import datetime
from radam import RAdam
import torch.nn as nn
import torch
import mlflow
import pandas as pd
import copy

class TrainModel:

    def __init__(self, config, model, feat):
        self.config = config
        self.model = model
        self.feat = feat
        self.consecutive_evals_no_improvement = 0

        self._initialize_device()
        self._initialize_loss()

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

    def _initialize_loss(self):
        self.criterion = nn.BCEWithLogitsLoss()

    def _initialize_optimizer(self, learning_rate):
        self.optimizer = RAdam(self.model.parameters(),
                               lr=learning_rate)

    @staticmethod
    def _format_time(elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train(self):

        self._initialize_optimizer(learning_rate=1e-5)

        # Initialize metrics
        self.best_loss = None

        # Freeze all Pretrained Layers for Stage 1 of Training
        for param in self.model.pretrained_model.parameters():
            param.requires_grad=False

        # Train Stage 1
        self._train()

        # Retrieve Best Model
        self.model = self.best_model

        self._initialize_optimizer(learning_rate=2e-6)

        # Un-Freeze all Layers
        for param in self.model.pretrained_model.parameters():
            param.requires_grad=True

        # Train Stage 2
        print ('Beginning stage 2 training')
        self._train()

    def _train(self):
        self.early_stop_criteria = False
        self.consecutive_evals_no_improvement = 0

        # Iterate through epochs
        for epoch_i in range(self.config.epochs):

            # Perform one full pass over the training set.
            print('----- EPOCH {:} / {:} -----'.format(epoch_i, self.config.epochs))
            print('Training...')

            # Initialize epoch variables
            t0 = time.time()
            total_loss = 0
            train_outputs = []

            # Iterate through the batch of training data
            for step, batch in enumerate(self.feat.train_data_loader):
                # Set the Model to Training Mode. Dropout and Batchnorm Layers perform Differently in train/eval_mode
                self.model.train()
                # Print Batch Progress
                if step % 20 == 0 and not step == 0:
                    elapsed = self._format_time(time.time() - t0)
                    print('Batch {:>5,} of {:>5,}. Elapsed: {:}.'.format(step, len(self.feat.train_data_loader), elapsed))

                # Send all the data to GPU
                b_input_ids = batch[0].long().to(self.device)
                b_input_masks = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                # Clear Previously collected gradient
                self.model.zero_grad()

                # Perform a forward pass
                outputs = self.model(b_input_ids,
                                     token_type_ids=None,
                                     attention_mask=None)

                # Append Sigmoids to Outputs
                train_outputs.append(outputs[1].detach().cpu().numpy())

                # Calculate loss and Accumulate Loss
                loss = self.criterion(outputs[0], b_labels)
                total_loss += loss.item()

                if self.config.mlflow_tracking: mlflow.log_metric('train_batch_loss', loss.item())

                # Perform a backward pass to calculate the gradients
                loss.backward()

                # Update Parameters
                self.optimizer.step()

                if step % self.config.evaluate_every == 0 and not step == 0:
                    # Initialize Validation metrics
                    t0 = time.time()
                    eval_loss = 0
                    nb_eval_steps = 0

                    # Put the model in eval mode
                    self.model.eval()
                    for batch in self.feat.test_data_loader:

                        # Add batch to GPU
                        batch = tuple(t.to(self.device) for t in batch)

                        # Unpack the inputs from our dataloader
                        b_input_ids, b_input_masks, b_labels = batch

                        # Don't calculate gradients on eval
                        with torch.no_grad():

                            # Forward pass, calculate logits
                            outputs = self.model(b_input_ids.long(),
                                                 token_type_ids=None,
                                                 attention_mask=None)

                            # Get the Logits Output by the model
                            logits = outputs[0]

                            # Move Logits and Labels to CPU
                            logits = logits.detach().cpu()
                            label_ids = b_labels.to('cpu')

                            # Calculate loss
                            loss = self.criterion(logits, label_ids)
                            eval_loss += loss.item()

                            if self.config.mlflow_tracking: mlflow.log_metric('eval_batch_loss', loss.item())

                    avg_eval_loss = eval_loss / len(self.feat.test_data_loader)
                    if self.config.mlflow_tracking: mlflow.log_metric('eval_avg_epoch_loss', avg_eval_loss)

                    print('Best Loss: {}'.format(self.best_loss))
                    print('Eval Loss: {}'.format(eval_loss))

                    if (self.best_loss is None) or (eval_loss < self.best_loss):
                        self.best_loss = eval_loss
                        self.consecutive_evals_no_improvement = 0

                        self.best_model = copy.deepcopy(self.model)

                        torch.save({
                            'epoch': epoch_i,
                            'model_state_dict': self.model.state_dict(),
                            'loss': self.best_loss
                        }, self.config.model_filename)

                        if self.config.mlflow_tracking: mlflow.log_artifact(self.config.model_filename)
                    else:
                        print('\n')
                        self.consecutive_evals_no_improvement += 1
                        print('Eval Loss increasing with most recent eval')
                        print('Number of consecutive increasing losses: ' + str(self.consecutive_evals_no_improvement))
                        if self.consecutive_evals_no_improvement >= self.config.consecutive_evals_no_improvement_to_stop:
                            print('Early Stopping...')
                            self.early_stop_criteria = True
                            if self.config.mlflow_tracking: mlflow.log_metric('epochs_before_convergence', epoch_i-1)
                    if self.early_stop_criteria:
                        break

                # return model, best_loss
                # TODO: Return appropriate best model

            # Calculate Average Training Loss and add to loss list
            avg_train_loss = total_loss / len(self.feat.train_data_loader)
            if self.config.mlflow_tracking: mlflow.log_metric('train_avg_epoch_loss', avg_train_loss)

            # Output Training Outputs
            sigmoid_outputs = [item for sublist in train_outputs for item in sublist]
            sigmoid_df = pd.DataFrame(sigmoid_outputs, columns=self.feat.label_cols)

            # Save Output Locally
            sigmoid_df.to_csv(self.config.output_filename)

            if self.config.mlflow_tracking: mlflow.log_artifact(self.config.output_filename)

            print('\n')
            print('Average Training Loss: {0:.2f}'.format(avg_train_loss))
            print('Training Epoch Took: {:}'.format(self._format_time(time.time() - t0)))

            # Evaluate data for one epoch
            if self.early_stop_criteria:
                break
