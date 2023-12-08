import torch
from torch.utils.data import Dataset
from transformers import AutoModel


class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        #bert-base-cased 768
        #bert-large-cased bert-large-uncased 1024
        #roberta-base-cased 768
        #biobert

        self.l1 = AutoModel.from_pretrained('models/bert-base-uncased')# BERT large
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.hidden_cls = torch.nn.Linear(768,768)
        self.hidden_parsing = torch.nn.Linear(768,768)
        self.hidden_den = torch.nn.Linear(768,768)
        self.hidden_vis = torch.nn.Linear(768,768)
        self.hidden_vis_pro = torch.nn.Linear(2048,768)
        self.hidden_all = torch.nn.Linear(768*2,768*2)
        self.before_classifier = torch.nn.Linear(768*2,128)
        self.pooling = torch.nn.MaxPool2d((2,1), stride=None)
        self.classifier = torch.nn.Linear(128, 4)

    def forward(self, input_ids, attention_mask, token_type_ids, char_density,char_number,visual_feature,bert_cls,parsing1,parsing2,visual):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]

        # BERT 768 BERT / large 1024
        
        # set different hidden layer, number of hidden units, regularization methods including bn and dropout
        
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)

        pooler = torch.cat((pooler.unsqueeze(1),bert_cls.unsqueeze(1)),1)
        pooler = self.pooling(pooler).squeeze(1)
        pooler = self.hidden_cls(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)

        visual = self.hidden_vis_pro(visual)
        visual = torch.nn.Tanh()(visual)
        visual = self.dropout(visual)

        visual = torch.cat((visual.unsqueeze(1),visual_feature.unsqueeze(1)),1)
        visual = self.pooling(visual).squeeze(1)
        visual = self.hidden_vis(visual)
        visual = torch.nn.Tanh()(visual)
        visual = self.dropout(visual)

        pooler = torch.cat((pooler,visual),1)
        pooler = self.hidden_all(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)

        pooler = self.before_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)

        output = self.classifier(pooler)
        return output


class TextBERTEncoder(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        # self.targets = self.data.label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float),
        }
    

class Feature_Extractor(torch.nn.Module):

    def __init__(self):
        super(Feature_Extractor, self).__init__()
        #bert-base-cased 768
        #bert-large-cased bert-large-uncased 1024
        #roberta-base-cased 768
        #biobert

        self.l1 = AutoModel.from_pretrained('bert-base-uncased')# BERT large
        self.pre_classifier = torch.nn.Linear(768, 768)

    def forward(self,input_ids,attention_mask,token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        
        return pooler
    

class DocGCN_Encoder(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len
        self.visual_feature = dataframe.near_visual_feature
        self.gcn_bert_base = dataframe.gcn_bert_predicted
        self.parsing1 = dataframe.level1_parse_emb
        self.parsing2 = dataframe.level2_parse_emb
        self.char_density = dataframe.gcn_near_char_density
        self.char_number = dataframe.gcn_near_char_number
        # self.pos_emb = dataframe.gcn_pos_emb
        self.visual = dataframe.visual_feature

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'gcn_bert_base': torch.tensor(self.gcn_bert_base[index],dtype=torch.float),
            'char_density': torch.tensor(self.char_density[index],dtype=torch.float),
            'char_number': torch.tensor(self.char_number[index],dtype=torch.float),
            'visual_feature': torch.tensor(self.visual_feature[index],dtype=torch.float),
            'parsing1': torch.tensor(self.parsing1[index],dtype=torch.float),
            'parsing2': torch.tensor(self.parsing2[index],dtype=torch.float),
            # 'pos_emb': torch.tensor(self.pos_emb[index],dtype=torch.float),
            'visual': torch.tensor(self.visual[index],dtype=torch.float),
        }
