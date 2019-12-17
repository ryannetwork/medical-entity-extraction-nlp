from collections import OrderedDict
import constants
from keras.preprocessing.sequence import pad_sequences
import glob
import math
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report,accuracy_score,f1_score
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import BertConfig,BertModel, BertTokenizer, AdamW, BertForTokenClassification
from tqdm import tqdm, trange


class MIMICBertNER:
    
    def __init__(self,cleaned_files_path):
        """
        Takes Training processed files or Testing processed files
        """
        self.cleaned_files_path = cleaned_files_path
        self.batch_num = 16
        self.tokenizer = BertTokenizer(vocab_file=constants.VOCAB_FILE, do_lower_case=True)
        self.bert_out_address = './saved_models/'
        if not os.path.exists(self.bert_out_address):
            os.makedirs(self.bert_out_address)

    def get_vectors(self,filename):
        vectors = []
        with open(filename,"rb") as f:
            try:
                while True:
                    x=pickle.load(f)
                    vectors = x
            except EOFError:
                pass
        return vectors

    def get_tokenized_texts_labels(self,sentences,labels):
        word_piece_labels = []
        i_inc = 0

        new_tokenized_texts = []

        #sentences = [s for s in new_sentences.split(" ")]

        for word_list,label in (zip(sentences,labels)):
            temp_label = []
            temp_token = []
            
            # Add [CLS] at the front 
            temp_label.append('[CLS]')
            temp_token.append('[CLS]')
            
            for word,lab in zip(word_list,label):
                token_list = self.tokenizer.tokenize(word)
                for m,token in enumerate(token_list):
                    temp_token.append(token)
                    if m==0:
                        temp_label.append(lab)
                    else:
                        temp_label.append('X')  
                        
            # Add [SEP] at the end
            temp_label.append('[SEP]')
            temp_token.append('[SEP]')
            
            new_tokenized_texts.append(temp_token)
            word_piece_labels.append(temp_label)
            
            i_inc +=1

        return new_tokenized_texts,word_piece_labels

    def get_all_sentences(self):
        #main_prefix = "/content/drive/Shared drives/BioNLP/project/"
        my_sentences = []
        my_labels = []
        target_vocab = []
        
        # file_name_regex = main_prefix + 'archive/medical_data/train_data/cleaned_files/*.dat'
        # file_prefix =  main_prefix + 'archive/medical_data/train_data/cleaned_files/'
        #l = [file_prefix + os.path.basename(x) for x in glob.glob(file_name_regex)]
        l = [os.path.join(self.cleaned_files_path,os.path.basename(x)) for x in glob.glob(self.cleaned_files_path)]

        for i in l:
            if 'label_dicts' not in i:
                g = get_vectors(i)
                my_sentences+=g[1]
                my_labels+=g[2]
                target_vocab+= g[0]
        
        target_entity_vocab = []
        for x in target_vocab:
            target_entity_vocab +=x['entity']
        
        return my_sentences,my_labels,target_entity_vocab

    def prepare_training_data(self):
        train_data = TensorDataset(self.tr_inputs, self.tr_masks, self.tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,num_workers=4,batch_size=self.batch_num)
        return train_dataloader

    def prepare_test_data(self):
        test_data = TensorDataset(self.val_inputs, self.val_masks, self.val_tags)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler,num_workers=4,batch_size=self.batch_num)
        return test_dataloader

    def get_inputs(self,mode):
        my_sentences,my_labels,target_vocab = self.get_all_sentences()
        new_sentences = [" ".join(x) for x in my_sentences]

        self.tag_map = {'b-problem':0,
                    'i-problem':1,
                    'b-test':2,
                    'i-test':3,
                    'b-treatment':4,
                    'i-treatment':5,
                    'O':6}

        reverse_tag_map = dict((v,k) for k,v in self.tag_map.items())

        #get all the labels
        new_labels = [[reverse_tag_map[j] for j in i] for i in my_labels]

        tags_vals = list(self.tag_map.keys())

        tags_vals.append('X')
        tags_vals.append('[CLS]')
        tags_vals.append('[SEP]')

        self.tag_map['X'] = 7
        self.tag_map['[CLS]'] = 8
        self.tag_map['[SEP]'] = 9

        tag2name={self.tag_map[key] : key for key in self.tag_map.keys()}

        new_sentences = [s.split(" ") for s in new_sentences]

        new_tokenized_texts,word_pieces_labels = self.get_tokenized_texts_labels(new_sentences,new_labels)

        MAX_LEN = 128

        #Cut and pad the token and label sequences to our desired length
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in new_tokenized_texts],
                                  maxlen = MAX_LEN, value = self.tag_map["O"],padding="post",
                                  dtype = "long",truncating = "post")

        tags = pad_sequences([[self.tag_map.get(l) for l in lab] for lab in word_piece_labels],
                             maxlen = MAX_LEN,value = self.tag_map["O"],padding = "post",
                            dtype="long",truncating="post")

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

        if mode == "train":
            tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                                        random_state=2018, test_size=0.2)
            tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                         random_state=2018, test_size=0.2)


            #converting the dataset to tensors for pytorch
            self.tr_inputs = torch.tensor(tr_inputs)
            self.val_inputs = torch.tensor(val_inputs)
            self.tr_tags = torch.tensor(tr_tags)
            self.val_tags = torch.tensor(val_tags)
            self.tr_masks = torch.tensor(tr_masks)
            self.val_masks = torch.tensor(val_masks)
        else:
            self.val_inputs = torch.tensor(input_ids)
            self.val_tags = torch.tensor(tags)


    def train(self):
        model = BertForTokenClassification.from_pretrained(constants.MIMIC_BERT_PRETRAINED_PATH, output_attentions=True,output_hidden_states=True,num_labels=len(self.tag_map))

        # #model = BertForTokenClassification.from_pretrained(model_file_address,num_labels=len(self.tag_map))
        if torch.cuda.is_available():
            model.cuda()
        n_gpu = torch.cuda.device_count()
        if n_gpu >1:
            model = torch.nn.DataParallel(model)
        epochs = 5
        max_grad_norm = 1.0
        batch_num = 16

        num_train_optimization_steps = int( math.ceil(len(self.tr_inputs) / batch_num) / 1) * epochs

        FULL_FINETUNING = True

        if FULL_FINETUNING:
            # Fine tune model all layer parameters
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            # Only fine tune classifier parameters
            param_optimizer = list(model.classifier.named_parameters()) 
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Batch_size
        # batch_size = 128
        # n_epochs = 5
        train_dataloader = self.prepare_training_data(self.tr_inputs, tr_masks, tr_tags)

        model.train()

        print("***** Running training *****")
        print("  Num examples = %d"%(len(self.tr_inputs)))
        print("  Batch size = %d"%(batch_num))
        print("  Num steps = %d"%(num_train_optimization_steps))
        for _ in trange(epochs,desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                
                # forward pass
                outputs = model(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask, labels=b_labels)
                # print("len",len(outputs),outputs)
                loss, scores = outputs[:2]
                # if n_gpu>1:
                #     # When multi gpu, average it
                #     loss = loss.mean()
                
                # backward pass
                loss.backward()
                
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                
                # update parameters
                optimizer.step()
                optimizer.zero_grad()
                
            # print train loss per epoch
            print("Train loss: {}".format(tr_loss/nb_tr_steps))

        return model

    def evaluate(self):
        model = BertForTokenClassification.from_pretrained(self.bert_out_address,num_labels=len(self.tag_map))

        valid_dataloader = self.prepare_test_data(self.val_inputs, self.val_masks, self.val_tags)

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []

        print("***** Running evaluation *****")
        print("  Num examples ={}".format(len(self.val_inputs)))
        print("  Batch size = {}".format(self.batch_num))
        for step, batch in enumerate(valid_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch
            
            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None,
                attention_mask=input_mask,)
                # For eval mode, the first result of outputs is logits
                logits = outputs[0] 
            
            # Get NER predict result
            logits = torch.argmax(F.log_softmax(logits,dim=2),dim=2)
            logits = logits.detach().cpu().numpy()
            
            # Get NER true result
            label_ids = label_ids.to('cpu').numpy()
            
            # Only predict the real word, mark=0, will not calculate
            input_mask = input_mask.to('cpu').numpy()
            
            # Compare the valuable predict result
            for i,mask in enumerate(input_mask):
                # Real one
                temp_1 = []
                # Predict one
                temp_2 = []
                
                for j, m in enumerate(mask):
                    # Mark=0, meaning its a pad word, dont compare
                    if m:
                        if tag2name[label_ids[i][j]] != "X" and tag2name[label_ids[i][j]] != "[CLS]" and tag2name[label_ids[i][j]] != "[SEP]" and tag2name[logits[i][j]] != "[SEP]": # Exclude the extra labels
                            temp_1.append(tag2name[label_ids[i][j]])
                            temp_2.append(tag2name[logits[i][j]])
                    else:
                        break
                    
                y_true.append(temp_1)
                y_pred.append(temp_2)

        # print("f1 score: %f"%(f1_score(y_true, y_pred)))
        # print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))

        # Get acc , recall, F1 result report

        report = classification_report(y_true, y_pred,digits=4)
        #print(report)

        # Save the report into file
        output_eval_file = os.path.join(self.bert_out_address, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            print("***** Eval results *****")
            print("\n%s"%(report))
            print("f1 score: %f"%(f1_score(y_true, y_pred)))
            print("Accuracy score: %f"%(accuracy_score(y_true, y_pred)))
            
            writer.write("f1 socre:\n")
            writer.write(str(f1_score(y_true, y_pred)))
            writer.write("\n\nAccuracy score:\n")
            writer.write(str(accuracy_score(y_true, y_pred)))
            writer.write("\n\n")  
            writer.write(report)
