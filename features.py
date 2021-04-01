import pandas as pd
import re
from transformers import AlbertTokenizer, BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from models import ModelConfig

class ESGData:

    def __init__(self, config):
        self.config = config
        self._read_in_data()

    def _read_in_data(self):
        self.train_df = pd.read_csv('train.csv')
        self.test_df = pd.read_csv('test.csv')

    def _generate_features(self, df):
        self.feature_cols = ['text']
        remove_cols = ['text']
        self.label_cols = list(df.columns)

        for remove_col in remove_cols:
            if remove_col in self.label_cols:
                label_cols = self.label_cols.remove(remove_col)

        # self.label_cols = label_cols
        self.num_labels = len(self.label_cols)
        df = df.dropna(subset=self.feature_cols)

        # Fill in all NAs in available data with 0s
        df = df.fillna(0)

        return df

    def get_tokens_and_attention_masks(self, input_sentences):
        # Initialize Tokenizer
        if self.config.model_type == 'ALBERT':
            tokenizer = AlbertTokenizer.from_pretrained(self.config.pretrained_base, do_lower_case=True)
        elif self.config.model_type == 'BERT':
            tokenizer = BertTokenizer.from_pretrained(self.config.pretrained_base, do_lower_case=True)

        # Initialize Objects for Tokens/Masks
        input_ids = []
        attention_masks = []

        for sent in input_sentences:
            tokenized = tokenizer.encode_plus(sent, max_length=self.config.max_seq_length, pad_to_max_length=True,
                                              truncation=True, padding='max_length')
            input_ids.append(tokenized['input_ids'])
            attention_masks.append(tokenized['attention_mask'])

        return input_ids, attention_masks

    def generate_data(self, df):
        self._generate_features(df)
        # df[self.feature_cols] = df[self.feature_cols].apply(lambda x: self.clean_text(x)).values
        return df

    def generate_torch_dataloaders(self, input_ids, attention_masks, labels=None):

        # Convert all data to torch tensors
        input_ids = torch.tensor(input_ids)

        labels = torch.tensor(labels).to(torch.float)
        attention_masks = torch.tensor(attention_masks)

        # Create Torch Dataloader for Data
        data = TensorDataset(input_ids, attention_masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data,
                                sampler=sampler,
                                batch_size=self.config.batch_size)

        return dataloader

    def execute(self):

        # Generate Train Data
        self.train_df = self.generate_data(self.train_df)
        self.train_labels = self.train_df[self.label_cols].values

        train_input_ids, train_attention_masks = self.get_tokens_and_attention_masks(
            self.train_df[self.feature_cols[0]])
        self.train_data_loader = self.generate_torch_dataloaders(input_ids=train_input_ids,
                                                                 attention_masks=train_attention_masks,
                                                                 labels=self.train_labels)

        # Generate Test Data
        self.test_df = self.generate_data(self.test_df)
        self.test_labels = self.test_df[self.label_cols].values
        test_input_ids, test_attention_masks = self.get_tokens_and_attention_masks(self.test_df[self.feature_cols[0]])
        self.test_data_loader = self.generate_torch_dataloaders(input_ids=test_input_ids,
                                                                attention_masks=test_attention_masks,
                                                                labels=self.test_labels)

if __name__ == '__main__':

    config = ModelConfig(model_type='BERT',
                         pretrained_base='bert-base-uncased',
                         layer_dict=None,
                         max_seq_length=64,
                         batch_size=16)

    esg = ESGData(config)
    esg.execute()

    print('Feature Preprocessing Successful.')
