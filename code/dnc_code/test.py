# Training DNC

import sys
import time
import torch
import random
import numpy as np

# torch.autograd.set_detect_anomaly(True) # Setting Anomaly Detection True for finding bad operations

####### Following function is adapted from the implemention by loudinthecloud on Github ##########
def random_seed():
    seed = int(time.time()*10000000)
    random.seed(seed)
    np.random.seed(int(seed/10000000))      # NumPy seed Range is 2**32 - 1 max
    torch.manual_seed(seed)
##########################################################################################################

def main():
    if len(sys.argv) > 4:
        if sys.argv[1] == "1":
            if sys.argv[2] == "GPU" and torch.cuda.is_available():  # Checking if GPU Request is given or not and availability of CUDA
                from tasks.babi_task_GPU import task_babi
            elif sys.argv[2] == "CPU":
                from tasks.babi_task import task_babi
            else:
                print("Please specify the run device (GPU/CPU)")
                exit()
            c_task = task_babi()                    # Initialization of the bAbI Task
            print("\nStarting bAbI Question Answering Task for DNC\n")
        elif sys.argv[1] == "2":
            if sys.argv[2] == "GPU" and torch.cuda.is_available():  # Checking if GPU Request is given or not and availability of CUDA
                from tasks.ner_task_GPU import task_NER
            elif sys.argv[2] == "CPU":
                from tasks.ner_task import task_NER
            else:
                print("Please specify the run device (GPU/CPU)")
                exit()
            c_task = task_NER()                    # Initialization of the bAbI Task
            print("\nStarting Medical NER Task for DNC\n")
        else:
            print("Unidentified task, please refer README file")
            exit()
    else:
        print("Incorrect Number of arguments")
        exit()

    epoch = sys.argv[3] # Last Epoch number till the model was trained (eg: 0)
    batch = sys.argv[4] # Last Batch Number till the model was trained (eg: 1000)
    batch_size = 1

    # Random Seed
    random_seed()

    c_task.init_dnc()
    c_task.init_loss()
    c_task.batch_size = batch_size
    c_task.load_model(2, epoch, batch)
    accuracy = c_task.test_model()

if __name__ == '__main__':
    main()