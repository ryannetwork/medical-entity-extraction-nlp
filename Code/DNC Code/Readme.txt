The code is written in Python 3.6 in Ubuntu 18.04 Operating System.

********************************************************************************

Libraries required:
NumPy
PyTorch

********************************************************************************

Train the model by writing following in the terminal:

"python3 train.py opt1 opt2"

---------------------------------------
opt1: 1). 1 for Copy_Task
      2). 2 for bAbI_Task

opt2: 1). GPU to run code on GPU
      2). CPU to run code normally
---------------------------------------

********************************************************************************

Test the model by writing following in the terminal:

"python3 test.py opt1 opt2 opt3 opt4"

---------------------------------------
opt1: 1). 1 for Copy_Task
      2). 2 for bAbI_Task

opt2: 1). GPU to run code on GPU
      2). CPU to run code normally

opt3: Last Epoch number till the model was trained (Not Applicable for Copy Task. Any value is fine)

Opt4: Last Batch Number till the model was trained 
---------------------------------------

********************************************************************************

Reference:
1). https://github.com/loudinthecloud/pytorch-ntm
2). Paper: Hybrid Computing Using a Neural Network with Dynamic External Memory
