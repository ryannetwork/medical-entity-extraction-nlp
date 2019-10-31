The code is written in Python 3.6 in Ubuntu 18.04 Operating System.

********************************************************************************

Libraries required:
NumPy
PyTorch

********************************************************************************

Train the model by writing following in the terminal:

"python3 train.py opt1 opt2"

---------------------------------------
opt1: 1). bAbI Task training
      2). Medical NER task training
      
opt1: 1). GPU to run code on GPU
      2). CPU to run code normally
---------------------------------------

********************************************************************************

Test the model by writing following in the terminal:

"python3 test.py opt1 opt2 opt3 opt4"

---------------------------------------
opt1: 1). bAbI Task training
      2). Medical NER task training

opt2: 1). GPU to run code on GPU
      2). CPU to run code normally

opt3: Last Epoch number till the model was trained

Opt4: Last Batch Number till the model was trained 
---------------------------------------

********************************************************************************

Reference:
1). https://github.com/loudinthecloud/pytorch-ntm
2). Paper: Robust and Scalable Differentiable Neural Computer for Question Answering
