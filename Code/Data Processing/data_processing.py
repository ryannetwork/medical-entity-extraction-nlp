import os
import numpy as np
import pickle

labelDict = {}     # Label Dictionary - Labels to Index
reverseDict = {}   # Inverse Label Dictionary - Index to Labels

def initialize_labels(s_path):     # Initializing label dictionaries for Labels->IDX and IDX->Labels
    # Using BIEOS labelling scheme
    labelDict['problem_b'] = 0     # Problem - Beginning 
    labelDict['problem_i'] = 1     # Problem - Inside
    labelDict['problem_e'] = 2     # Problem - End
    labelDict['problem_s'] = 3     # Problem - Single
    labelDict['test_b'] = 4        # Test - Beginning
    labelDict['test_i'] = 5        # Test - Inside
    labelDict['test_e'] = 6        # Test - End
    labelDict['test_s'] = 7        # Test - Single
    labelDict['treatment_b'] = 8   # Treatment - Beginning
    labelDict['treatment_i'] = 9   # Treatment - Inside
    labelDict['treatment_e'] = 10  # Treatment - End
    labelDict['treatment_s'] = 11  # Treatment - Single
    labelDict['o'] = 12            # Outside Token

    # Making Inverse Label Dictionary
    for k in labelDict.keys():
        reverseDict[labelDict[k]] = k
    
    # Saving the diictionaries into a file
    save_data([labelDict, reverseDict], os.path.join(s_path, "label_dicts.dat"))

def parse_concepts(file_path):      # Parses the concept file to extract concepts and labels
    conceptList = []                # Stores all the Concept in the File

    f = open(file_path)             # Opening and reading a concept file
    content = f.readlines()         # Reading all the lines in the concept file
    f.close()                       # Closing the concept file

    for x in content:               # Reading each line in the concept file
        dic = {}

        # Cleaning and extracting the entities, labels and their positions in the corresponding medical summaries
        x = x.strip().split('||')

        temp1, label = x[0].split(' '), x[1].split('=')[1][1:-1]

        temp1[0] = temp1[0][3:]
        temp1[-3] = temp1[-3][0:-1]
        entity = temp1[0:-2]

        if len(entity) > 1:
            lab = ['i']*len(entity)
            lab[0] = 'b'
            lab[-1] = 'e'
            lab = [label+"_"+l for l in lab]
        elif len(entity) == 1:
            lab = [label+"_"+"s"]
        else:
            print("Data in File: " + file_path + ", not in expected format..")
            exit()

        noLab = [labelDict[l] for l in lab]
        sLine, sCol = int(temp1[-2].split(":")[0]), int(temp1[-2].split(":")[1])
        eLine, eCol = int(temp1[-1].split(":")[0]), int(temp1[-1].split(":")[1])
        
        '''
        # Printing the information
        print("------------------------------------------------------------")
        print("Entity: " + str(entity))
        print("Entity Label: " + label)
        print("Labels - BIEOS form: " + str(lab))
        print("Labels  Index: " + str(noLab))
        print("Start Line: " + str(sLine) + ", Start Column: " + str(sCol))
        print("End Line: " + str(eLine) + ", End Column: " + str(eCol))
        print("------------------------------------------------------------")
        '''

        # Storing the information as a dictionary
        dic['entity'] = entity      # Entity Name (In the form of list of words)
        dic['label'] = label        # Common Label
        dic['BIEOS_labels'] = lab   # List of BIEOS label for each word
        dic['label_index'] = noLab  # Labels in the index form
        dic['start_line'] = sLine   # Start line of the concept in the corresponding text summaries
        dic['start_word_no'] = sCol # Starting word number of the concept in the corresponding start line
        dic['end_line'] = eLine     # End line of the concept in the corresponding text summaries
        dic['end_word_no'] = eCol   # Ending word number of the concept in the corresponding end line

        # Appending the concept dictionary to the list
        conceptList.append(dic)

    return conceptList  # Returning the all the concepts in the current file in the form of dictionary list

def parse_summary(file_path):       # Parses the Text summaries
    file_lines = []                 # Stores the lins of files in the list form
    tags = []                       # Stores corresponding labels for each word in the file (Default label: 'o' [Outside])
    # counter = 1                   # Temporary variable

    f = open(file_path)             # Opening and reading a concept file
    content = f.readlines()         # Reading all the lines in the concept file
    f.close()

    for x in content:
        file_lines.append(x.strip().split(" "))     # Appending the lines in the list
        tags.append([12]*len(file_lines[-1]))               # Assigining the default labels to all the words in a line
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

def modify_labels(conceptList, tags):   # Modifies he default labels to each word with the true labels from the concept files
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

def print_data(file, file_lines, tags):       # Prints the given data
    counter = 1

    print("\n************ Printing details of the file: " + file + " ************\n")
    for x in file_lines:
        print("------------------------------------------------------------")
        print("File Lines No: " + str(counter))
        print(x)
        print("\nCorresponding labels:")
        print([reverseDict[i] for i in tags[counter-1]])
        print("\nCorresponding Label Indices:")
        print(tags[counter-1])
        print("------------------------------------------------------------")
        counter += 1

def save_data(obj_list, s_path):                # Saves the file into the binary file using Pickle
    pickle.dump(tuple(obj_list), open(s_path,'wb'))

def process_data(c_path, t_path, s_path):      # Read all the concept files to get concepts and labels, proces them and save them
    for f in os.listdir(t_path):
        f1 = f.split('.')[0] + ".con"
        if os.path.isfile(os.path.join(c_path, f1)):
            conceptList = parse_concepts(os.path.join(c_path, f1))      # Parsing concepts and labels from the corresponding concept file
            file_lines, tags = parse_summary(os.path.join(t_path, f))   # Parses the document summaries to get the written notes
            tags = modify_labels(conceptList, tags)                     # Modifies he default labels to each word with the true labels from the concept files
            save_data([conceptList, file_lines, tags], os.path.join(s_path, f.split('.')[0]+".dat"))          # Saving the objects into a file
            # print_data(f, file_lines, tags)                           # Printing the details

if __name__ == '__main__':

    # File paths
    save_path = "../../Medical Data/Training Data/cleaned_files"
    concept_path = "../../Medical Data/Training Data/training_data/partners/concept"
    text_path = "../../Medical Data/Training Data/training_data/partners/txt"

    initialize_labels(save_path)                        # Initializing and saving the label dictionaries
    process_data(concept_path, text_path, save_path)    # Processing the data