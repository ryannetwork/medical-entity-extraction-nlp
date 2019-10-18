# Final DNC version packaging all the modules
import torch
from torch import nn
from .memory import memory_unit
from .processor import processor
from .controller import backward_controller

class DNC_Module(nn.Module):

    def __init__(self, num_inputs, num_outputs, controller_size, controller_layers, num_read_heads, num_write_heads, N, M):

        # Params:
        # num_inputs : Size of input data
        # num_outputs : Size of output data
        # controller_size : Size of LSTM Controller output/state
        # controller_layers : Number of layers in LSTM Network
        # num_read_heads : Number of Read heads to be created
        # num_write_heads : Number of Write heads to be created
        # N : Number of memory cells
        # M : Size of Each memory cell

        super(DNC_Module, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.N = N
        self.M = M

        # Creating NTM modules
        self.memory = memory_unit(self.N, self.M)
        self.processor = processor(self.num_inputs, self.num_outputs, self.M, self.N, num_read_heads, num_write_heads, controller_size, controller_layers)

        # Creating the Reverse Controller
        self.bController = backward_controller(self.num_inputs, controller_size, controller_layers)

        # Sending modules to CUDA cores
        self.memory.cuda()
        self.processor.cuda()
        self.bController.cuda()

    def initialization(self, batch_size):   # Initializing all the Modules
        self.batch_size = batch_size
        self.memory.reset_memory(batch_size)
        self.previous_state = self.processor.create_new_state(batch_size)
        self.previous_backward_states = self.bController.create_hidden_state(batch_size)

    def backward_prediction(self, X):       # Giving the input to the Backward Controller for making DNC Bi-Directional
        # X dim: (seq_len x batch_size x self.num_inputs)
        output, _ = self.bController(X, self.previous_backward_states) # Output dim: (seq_len x batch x controller_size)
        return output                                                  # Use embiddings from last to first (reverse way)

    def forward(self, X=None, backward_embeddings=None):
        if X is None:
            X = torch.zeros(self.batch_size, self.num_inputs).cuda()
        out, self.previous_state = self.processor(X, backward_embeddings, self.previous_state, self.memory)
        return out, self.previous_state
    
    '''
    def calculate_num_params(self):     # This maybe for model statistics. Adapted from GitHub Implementation
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
    '''