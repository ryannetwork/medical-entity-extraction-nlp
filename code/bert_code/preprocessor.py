import os
import numpy as np
import pickle
import re

class PreProcessor(object):
    def __init__(self,concept_path,text_path,save_path):
        """
        Training path - 
        concept_path = "../dnc_code/medical_data/train_data/concept"
    text_path = "../dnc_code/medical_data/train_data/txt"
        Testing path
        concept_path = "../dnc_code/medical_data/test_data/concept"
    text_path = "../dnc_code/medical_data/test_data/txt"
        """
        self.concept_path = concept_path
        self.text_path = text_path
        self.save_path = save_path

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

    def save_data(self,obj_list, s_path):                # Saves the file into the binary file using Pickle
        pickle.dump(tuple(obj_list), open(s_path,'wb'))

    def process_data(self,c_path, t_path, s_path):      # Read all the concept files to get concepts and labels, proces them and save them
        for f in os.listdir(t_path):
            f1 = f.split('.')[0] + ".con"
            if os.path.isfile(os.path.join(c_path, f1)):
                conceptList = self.parse_concepts(os.path.join(c_path, f1))      # Parsing concepts and labels from the corresponding concept file
                file_lines, tags = self.parse_summary(os.path.join(t_path, f))   # Parses the document summaries to get the written notes
                tags = self.modify_labels(conceptList, tags)                     # Modifies he default labels to each word with the true labels from the concept files
                self.save_data([conceptList, file_lines, tags], os.path.join(s_path, f.split('.')[0]+".dat"))          # Saving the objects into a file
                # print_data(f, file_lines, tags)                           # Printing the details


    def pre_process(self):
        """
        Any other preprocessing needed can be called from pre_process method.
        """
        # self.remove_duplicates_from_visual_descriptor_dataset()
        # self.rename_image_ids_from_visual_descriptor_dataset()
        # self.add_missing_objects_to_dataset()
        # self.transform_graph_file_to_dict_graph()
        # self.transform_edgelist_to_list_of_list_graph() -> not used

        self.initialize_labels()
        self.process_data(self.concept_path,self.text_path, self.save_path)


