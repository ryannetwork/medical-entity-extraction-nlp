# Named Entity Recognition on Medical Data (BIO Tagging)
# Bio-Word2Vec Embeddings Source and Reference: https://github.com/ncbi-nlp/BioWordVec

import os
import re
import torch
import pickle
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import random

from DNC_GPU.dnc import DNC_Module  # Importing DNC Implementation

class task_NER():

    def __init__(self):
        self.name = "NER_task_bio"

        # Controller Params
        self.controller_size = 128
        self.controller_layers = 1

        # Head Params
        self.num_read_heads = 2
        self.num_write_heads = 2

        # Processor Params
        self.num_inputs = 200           # Length of Embeddings
        self.num_outputs = 13           # Class size

        # Memory Params
        self.memory_N = 128
        self.memory_M = 128

        # Training Params
        self.num_batches = -1
        self.save_batch = 500           # Saving model after every save_batch number of batches
        self.batch_size = 10
        self.num_epoch = 1

        # Optimizer Params
        self.adam_lr = 1e-4
        self.adam_betas = (0.9, 0.999)
        self.adam_eps = 1e-8

        # Handles
        self.machine = None
        self.loss = None
        self.optimizer = None

        # Class Dictionaries
        self.labelDict = None         # Label Dictionary - Labels to Index
        self.reverseDict = None       # Inverse Label Dictionary - Index to Labels

        # File Paths
        self.concept_path_train = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/train_data/concept"  # Path to train concept files
        self.text_path_train = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/train_data/txt"         # Path to train text summaries
        self.concept_path_test = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/test_data/concept"    # Path to test concept files
        self.text_path_test = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/test_data/txt"           # Path to test text summaries
        self.save_path = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/cleaned_files"                # Save path
        self.embed_dic_path = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/embeddings/bio_embedding_dictionary.dat"  # Word2Vec embeddings Dictionary path
        self.random_vec = "/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/medical_data/embeddings/random_vec.dat"   # Path to random embedding (Used to create new vectors)

        # Miscellaneous
        self.padding_symbol = np.full((self.num_inputs), 0.01)        # Padding symbol embedding

    def get_task_name(self):
        return self.name

    def init_dnc(self):
        self.machine = DNC_Module(self.num_inputs, self.num_outputs, self.controller_size, self.controller_layers, self.num_read_heads, self.num_write_heads, self.memory_N, self.memory_M)
        self.machine.cuda()     # Enabling GPU

    def init_loss(self):
        self.loss = nn.CrossEntropyLoss(reduction = 'mean').cuda()  # Cross Entropy Loss -> Softmax Activation + Cross Entropy Loss

    def init_optimizer(self):
        self.optimizer = optim.Adam(self.machine.parameters(), lr = self.adam_lr, betas = self.adam_betas, eps = self.adam_eps)

    def calc_loss(self, Y_pred, Y):
        # Y: dim -> (sequence_len x batch_size)
        # Y_pred: dim -> (sequence_len x batch_size x num_outputs)
        loss_vec = torch.empty(Y.shape[0], dtype=torch.float32).cuda()
        for i in range(Y_pred.shape[0]):
            loss_vec[i] = self.loss(Y_pred[i], Y[i])
        return torch.mean(loss_vec)

    def calc_cost(self, Y_pred, Y):       # Calculates % Cost   # Needs rework. Consider whole entity instead of words
        # Y: dim -> (sequence_len x batch_size)
        # Y_pred: dim -> (sequence_len x batch_size x sequence_width)
        return torch.sum(((F.softmax(Y_pred, dim=2).max(2)[1]) == Y).type(torch.long)).item(), Y.shape[0]*Y.shape[1]

    def print_word(self, token_class):         # Prints the Class name from Class number
        word = self.reverseDict[token_class]
        print(word + "\n")

    def clip_grads(self):       # Clipping gradients for stability
        """Gradient clipping to the range [10, 10]."""
        parameters = list(filter(lambda p: p.grad is not None, self.machine.parameters()))
        for p in parameters:
            p.grad.data.clamp_(-10, 10)

    def initialize_labels(self):            # Initializing label dictionaries for Labels->IDX and IDX->Labels
        self.labelDict = {}                 # Label Dictionary - Labels to Index
        self.reverseDict = {}               # Inverse Label Dictionary - Index to Labels

        # Using BIEOS labelling scheme
        self.labelDict['b-problem'] = 0     # Problem - Beginning 
        self.labelDict['i-problem'] = 1     # Problem - Inside
        self.labelDict['b-test'] = 2        # Test - Beginning
        self.labelDict['i-test'] = 3        # Test - Inside
        self.labelDict['b-treatment'] = 4   # Treatment - Beginning
        self.labelDict['i-treatment'] = 5   # Treatment - Inside
        self.labelDict['o'] = 6             # Outside Token

        # Making Inverse Label Dictionary
        for k in self.labelDict.keys():
            self.reverseDict[self.labelDict[k]] = k

        # Saving the diictionaries into a file
        self.save_data([self.labelDict, self.reverseDict], os.path.join(self.save_path, "label_dicts_bio.dat"))

    def parse_concepts(self, file_path):    # Parses the concept file to extract concepts and labels
        conceptList = []                    # Stores all the Concept in the File

        f = open(file_path)                 # Opening and reading a concept file
        content = f.readlines()             # Reading all the lines in the concept file
        f.close()                           # Closing the concept file

        for x in content:                   # Reading each line in the concept file
            dic = {}

            # Cleaning and extracting the entities, labels and their positions in the corresponding medical summaries
            x = re.sub('\n', ' ', x)
            x = re.sub(r'\ +', ' ', x)
            x = x.strip().split('||')

            temp1, label = x[0].split(' '), x[1].split('=')[1][1:-1]

            temp1[0] = temp1[0][3:]
            temp1[-3] = temp1[-3][0:-1]
            entity = temp1[0:-2]

            if len(entity) >= 1:
                lab = ['i']*len(entity)
                lab[0] = 'b'
                lab = [l+"-"+label for l in lab]
            else:
                print("Data in File: " + file_path + ", not in expected format..")
                exit()

            noLab = [self.labelDict[l] for l in lab]
            sLine, sCol = int(temp1[-2].split(":")[0]), int(temp1[-2].split(":")[1])
            eLine, eCol = int(temp1[-1].split(":")[0]), int(temp1[-1].split(":")[1])
            
            '''
            # Printing the information
            print("------------------------------------------------------------")
            print("Entity: " + str(entity))
            print("Entity Label: " + label)
            print("Labels - BIO form: " + str(lab))
            print("Labels  Index: " + str(noLab))
            print("Start Line: " + str(sLine) + ", Start Column: " + str(sCol))
            print("End Line: " + str(eLine) + ", End Column: " + str(eCol))
            print("------------------------------------------------------------")
            '''

            # Storing the information as a dictionary
            dic['entity'] = entity      # Entity Name (In the form of list of words)
            dic['label'] = label        # Common Label
            dic['BIO_labels'] = lab     # List of BIO labels for each word
            dic['label_index'] = noLab  # Labels in the index form
            dic['start_line'] = sLine   # Start line of the concept in the corresponding text summaries
            dic['start_word_no'] = sCol # Starting word number of the concept in the corresponding start line
            dic['end_line'] = eLine     # End line of the concept in the corresponding text summaries
            dic['end_word_no'] = eCol   # Ending word number of the concept in the corresponding end line

            # Appending the concept dictionary to the list
            conceptList.append(dic)

        return conceptList  # Returning the all the concepts in the current file in the form of dictionary list

    def parse_summary(self, file_path):         # Parses the Text summaries
        file_lines = []                         # Stores the lins of files in the list form
        tags = []                               # Stores corresponding labels for each word in the file (Default label: 'o' [Outside])
        default_label = len(self.labelDict)-1   # default_label is "7" (Corresponding to 'Other' entity) 
        # counter = 1                           # Temporary variable used during print

        f = open(file_path)             # Opening and reading a concept file
        content = f.readlines()         # Reading all the lines in the concept file
        f.close()

        for x in content:
            x = re.sub('\n', ' ', x)
            x = re.sub(r'\ +', ' ', x)
            file_lines.append(x.strip().split(" "))             # Spliting the lines into word list and Appending each of them in the file list
            tags.append([default_label]*len(file_lines[-1]))    # Assigining the default_label to all the words in a line
            '''
            # Printing the information
            print("------------------------------------------------------------")
            print("File Lines No: " + str(counter))
            print(file_lines[-1])
            print("\nCorresponding labels:")
            print(tags[-1])
            print("------------------------------------------------------------")
            counter += 1
            '''
            assert len(tags[-1]) == len(file_lines[-1]), "Line length is not matching labels length..."    # Sanity Check
        return file_lines, tags

    def modify_labels(self, conceptList, tags):         # Modifies the default labels of each word in text files with the true labels from the concept files
        for e in conceptList:                           # Iterating over all the dictionary elements in the Concept List
            if e['start_line'] == e['end_line']:        # Checking whether concept is spanning over a single line or multiple line in the summary
                tags[e['start_line']-1][e['start_word_no']:e['end_word_no']+1] = e['label_index'][:]
            else:
                start = e['start_line']
                end = e['end_line']
                beg = 0
                for i in range(start, end+1):           # Distributing labels over multiple lines in the text summaries
                    if i == start:
                        tags[i-1][e['start_word_no']:] = e['label_index'][0:len(tags[i-1])-e['start_word_no']]
                        beg = len(tags[i-1])-e['start_word_no']
                    elif i == end:
                        tags[i-1][0:e['end_word_no']+1] = e['label_index'][beg:]
                    else:
                        tags[i-1][:] = e['label_index'][beg:beg+len(tags[i-1])]
                        beg = beg+len(tags[i-1])
        return tags

    def print_data(self, file, file_lines, tags):       # Prints the given data
        counter = 1

        print("\n************ Printing details of the file: " + file + " ************\n")
        for x in file_lines:
            print("------------------------------------------------------------")
            print("File Lines No: " + str(counter))
            print(x)
            print("\nCorresponding labels:")
            print([self.reverseDict[i] for i in tags[counter-1]])
            print("\nCorresponding Label Indices:")
            print(tags[counter-1])
            print("------------------------------------------------------------")
            counter += 1

    def save_data(self, obj_list, s_path):              # Saves the file into the binary file using Pickle
        # Note: The 'obj_list' must be a list and none other than that
        pickle.dump(tuple(obj_list), open(s_path,'wb'))

    def acquire_data(self, task):   # Read all the concept files to get concepts and labels, proces them and save them
        data = {}                   # Dictionary to store all the data objects (conceptList, file_lines, tags) each indexed by file name

        if task == 'train':         # Determining the task type to assign the data path accordingly
            t_path = self.text_path_train
            c_path = self.concept_path_train
        else:
            t_path = self.text_path_test
            c_path = self.concept_path_test

        for f in os.listdir(t_path):
            f1 = f.split('.')[0] + ".con"
            if os.path.isfile(os.path.join(c_path, f1)):
                conceptList = self.parse_concepts(os.path.join(c_path, f1))         # Parsing concepts and labels from the corresponding concept file
                file_lines, tags = self.parse_summary(os.path.join(t_path, f))      # Parses the document summaries to get the written notes
                tags = self.modify_labels(conceptList, tags)                        # Modifies he default labels to each word with the true labels from the concept files
                data[f1] = [conceptList, file_lines, tags]                          # Storing each object in dictionary
                # self.print_data(f, file_lines, tags)                              # Printing the details
        return data

    def structure_data(self, data_dict):        # Structures the data in proper trainable form
        final_line_list = []                    # Stores words of all the files in separate sub-lists
        final_tag_list = []                     # Stores tags of all the files in separate sub-lists

        for k in data_dict.keys():              # Extracting data from each pre-processed file in dictionary
            file_lines = data_dict[k][1]        # Extracting story
            tags = data_dict[k][2]              # Extracting corresponding labels

            # Creating empty lists
            temp1 = []
            temp2 = []

            # Merging all the lines in file into a single list. Same for corresponding labels
            for i in range(len(file_lines)):
                temp1.extend(file_lines[i])
                temp2.extend(tags[i])
            
            assert len(temp1) == len(temp2), "Word length not matching Label length for story in " + str(k)     # Sanity Check

            final_line_list.append(temp1)
            final_tag_list.append(temp2)
        
        assert len(final_line_list) == len(final_tag_list), "Number of stories not matching number of labels list"  # Sanity Check
        return final_line_list, final_tag_list
    
    def padding(self, line_list, tag_list):     # Pads stories with padding symbol to make them of same length 
        diff = 0
        max_len = 0
        outside_class = len(self.labelDict)-1   # Classifying padding symbol as "outside" term

        # Calculating Max Summary Length
        for i in range(len(line_list)):
            if len(line_list[i])>max_len:
                max_len = len(line_list[i])

        for i in range(len(line_list)):
            diff = max_len - len(line_list[i])
            line_list[i].extend([self.padding_symbol]*diff)
            tag_list[i].extend([outside_class]*diff)
            assert (len(line_list[i]) == max_len) and (len(line_list[i]) == len(tag_list[i])), "Padding unsuccessful"    # Sanity check
        return np.asarray(line_list), np.asarray(tag_list)      # Making NumPy array of size (batch_size x story_length x word size) and (batch_size x story_length x 1) respectively

    def embed_input(self, line_list):       # Converts words to vector embeddings
        final_list = [] # Stores embedded words
        summary = None  # Temp variable
        word = None     # Temp variable
        temp = None     # Temp variable

        embed_dic = pickle.load(open(self.embed_dic_path, 'rb'))  # Loading word2vec dictionary using Pickle
        r_embed = pickle.load(open(self.random_vec, 'rb'))        # Loading Random embedding

        for i in range(len(line_list)):     # Iterating over all the summaries
            summary = line_list[i]
            final_list.append([])           # Reserving space for curent summary

            for j in range(len(summary)):
                word = summary[j].lower()
                if word in embed_dic:       # Checking for existence of word in dictionary
                    final_list[-1].append(embed_dic[word])
                else:
                    temp = r_embed[:]       # Copying the values of the list
                    random.shuffle(temp)    # Randomly shuffling the word embedding to make it unique
                    temp = np.asarray(temp, dtype=np.float32)   # Converting to NumPy array
                    final_list[-1].append(temp)
        return final_list

    def prepare_data(self, task='train'):       # Preparing all the data necessary
        line_list, tag_list = None, None

        '''
        line_list is the list of rows, where each row is a list of all the words in a medical summary
        Similar is the case for tag_list, except, it stores labels for each words
        '''

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)            # Creating a new directory if it does not exist else reading previously saved data
        
        if not os.path.exists(os.path.join(self.save_path, "label_dicts_bio.dat")):
            self.initialize_labels()                                                                                        # Initialize label to index dictionaries
        else:
            self.labelDict, self.reverseDict = pickle.load(open(os.path.join(self.save_path, "label_dicts_bio.dat"), 'rb')) # Loading Label dictionaries
        
        if not os.path.exists(os.path.join(self.save_path, "object_dict_bio_"+str(task)+".dat")):
            data_dict = self.acquire_data(task)                                                                             # Read data from file
            line_list, tag_list = self.structure_data(data_dict)                                                            # Structures the data into proper form
            line_list = self.embed_input(line_list)                                                                         # Embeds input data (words) into embeddings
            self.save_data([line_list, tag_list], os.path.join(self.save_path, "object_dict_bio_"+str(task)+".dat"))
        else:
            line_list, tag_list = pickle.load(open(os.path.join(self.save_path, "object_dict_bio_"+str(task)+".dat"), 'rb'))    # Loading Data dictionary
        return line_list, tag_list

    def get_data(self, task='train'):
        line_list, tag_list = self.prepare_data(task)

        # Shuffling stories
        story_idx = list(range(0, len(line_list)))
        random.shuffle(story_idx)

        num_batch = int(len(story_idx)/self.batch_size)
        self.num_batches = num_batch

        # Out Data
        x_out = []
        y_out = []

        counter = 1

        for i in story_idx:
            if num_batch<=0:
                break

            x_out.append(line_list[i])
            y_out.append(tag_list[i])

            if counter % self.batch_size == 0:
                counter = 0

                # Padding and converting labels to one hot vectors
                x_out_pad, y_out_pad = self.padding(x_out, y_out)
                x_out_array = torch.tensor(x_out_pad.swapaxes(0, 1), dtype=torch.float32)                       # Converting from (batch_size x story_length x word size) to (story_length x batch_size x word size)
                y_out_array = torch.tensor(y_out_pad.swapaxes(0, 1), dtype=torch.long)     # Converting from (batch_size x story_length x 1) to (story_length x batch_size x 1)

                x_out = []
                y_out = []
                num_batch -= 1

                yield (self.num_batches - num_batch), x_out_array, y_out_array
            counter += 1

    def train_model(self):
        # Here, the model is optimized using Cross Entropy Loss, however, it is evaluated using Number of error bits in predction and actual labels (cost)
        loss_list = []
        seq_length = []
        last_batch = 0

        for j in range(self.num_epoch):
            for batch_num, X, Y in self.get_data(task='train'):
                self.optimizer.zero_grad()                      # Making old gradients zero before calculating the fresh ones
                self.machine.initialization(self.batch_size)    # Initializing states
                Y_out = torch.empty((X.shape[0], X.shape[1], self.num_outputs), dtype=torch.float32).cuda()    # dim: (seq_len x batch_size x num_output)

                # Feeding the DNC network all the data first and then predicting output
                # by giving zero vector as input and previous read states and hidden vector
                # and thus training vector this way to give outputs matching the labels

                X, Y = X.cuda(), Y.cuda()

                embeddings = self.machine.backward_prediction(X)        # Creating embeddings from data for backward calculation
                temp_size = X.shape[0]

                for i in range(temp_size):
                    Y_out[i, :, :], _ = self.machine(X[i], embeddings[temp_size-i-1])   # Passing Embeddings from backwards

                loss = self.calc_loss(Y_out, Y)
                loss.backward()
                self.clip_grads()
                self.optimizer.step()

                corr, tot = self.calc_cost(Y_out.cpu(), Y.cpu())

                loss_list += [loss.item()]
                seq_length += [Y.shape[0]]

                if (batch_num % self.save_batch) == 0:
                    self.save_model(j, batch_num)

                last_batch = batch_num
                print("Epoch: " + str(j) + "/" + str(self.num_epoch) + ", Batch: " + str(batch_num) + "/" + str(self.num_batches) + ", Loss: " + str(loss.item()) + ", Batch Accuracy: " + str((float(corr)/float(tot))*100.0) + " %")
            self.save_model(j, last_batch)

    def test_model(self):   # Testing the model
        correct = 0
        total = 0
        print("\n")

        for batch_num, X, Y in self.get_data(task='test'):
            self.machine.initialization(self.batch_size)    # Initializing states
            Y_out = torch.empty((X.shape[0], X.shape[1], self.num_outputs), dtype=torch.float32).cuda()    # dim: (seq_len x batch_size x num_output)

            # Feeding the DNC network all the data first and then predicting output
            # by giving zero vector as input and previous read states and hidden vector
            # and thus training vector this way to give outputs matching the labels

            X = X.cuda()

            embeddings = self.machine.backward_prediction(X)        # Creating embeddings from data for backward calculation
            temp_size = X.shape[0]

            for i in range(temp_size):
                Y_out[i, :, :], _ = self.machine(X[i], embeddings[temp_size-i-1])

            corr, tot = self.calc_cost(Y_out.cpu(), Y)

            correct += corr
            total += tot
            print("Test Example " + str(batch_num) + "/" + str(self.num_batches) + " processed, Batch Accuracy: " + str((float(corr)/float(tot))*100.0) + " %")
        
        accuracy = (float(correct)/float(total))*100.0
        print("\nOverall Accuracy: " + str(accuracy) + " %")
        return accuracy         # in %

    def save_model(self, curr_epoch, curr_batch):
        # Here 'start_epoch' and 'start_batch' params below are the 'epoch' and 'batch' number from which to start training after next model loading
        # Note: It is recommended to start from the 'start_epoch' and not 'start_epoch' + 'start_batch', because batches are formed randomly
        if not os.path.exists("Saved_Models/" + self.name):
            os.mkdir("Saved_Models/" + self.name)
        state_dic = {'task_name': self.name, 'start_epoch': curr_epoch + 1, 'start_batch': curr_batch + 1, 'state_dict': self.machine.state_dict(), 'optimizer_dic' : self.optimizer.state_dict()}
        filename = "Saved_Models/" + self.name + "/" + self.name + "_" + str(curr_epoch) + "_" + str(curr_batch) + "_saved_model.pth.tar"
        torch.save(state_dic, filename)

    def load_model(self, option, epoch, batch):
        path = "Saved_Models/" + self.name + "/" + self.name + "_" + str(epoch) + "_" + str(batch) + "_saved_model.pth.tar"
        if option == 1:             # Loading for training
            checkpoint = torch.load(path)
            self.machine.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_dic'])
        else:                       # Loading for testing
            checkpoint = torch.load(path)
            self.machine.load_state_dict(checkpoint['state_dict'])
            self.machine.eval()