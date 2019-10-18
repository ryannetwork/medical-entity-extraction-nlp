# Read and Write Head controller based on LSTM.
# Note : Derived from GitHub user loudinthecloud's NTM implementation

import torch
from torch import nn
from torch.nn import Parameter
import numpy as np

class controller(nn.Module):    # LSTM Controller
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(controller, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm_network = nn.LSTM(input_size = self.num_inputs, hidden_size = self.num_outputs, num_layers = self.num_layers)

        # Parameters of the LSTM. Hidden state serves as the output of our network
        self.h_init = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)   # Hidden state initialization
        self.c_init = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)   # C variable initialization

        # Initialization of the LSTM parameters.
        for p in self.lstm_network.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))  # I don't know why we multiplied 5
                nn.init.uniform_(p, -stdev, stdev)

    def create_hidden_state(self, batch_size):  # Output : (num_layers x batch_size x num_outputs)
        h = self.h_init.clone().repeat(1, batch_size, 1)
        c = self.c_init.clone().repeat(1, batch_size, 1)
        return h, c

    def network_size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, inp, prev_state):
        inp = inp.unsqueeze(0)                              # inp dimension after unsqueeze : (1 x inp.shape)
        output, state = self.lstm_network(inp, prev_state)  # Input to LSTM must be of shape (seq_len x batch_size x input_size) in Pytorch. Here, seq_len = 1
        return output.squeeze(0), state

class backward_controller(nn.Module):   # Backward LSTM to make DNC Bi-Directional
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(backward_controller, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm_network = nn.LSTM(input_size = self.num_inputs, hidden_size = self.num_outputs, num_layers = self.num_layers)

        # Parameters of the LSTM. Hidden state serves as the output of our network
        self.h_init = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)   # Hidden state initialization
        self.c_init = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)   # C variable initialization

        # Initialization of the LSTM parameters.
        for p in self.lstm_network.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))  # I don't know why we multiplied 5
                nn.init.uniform_(p, -stdev, stdev)

    def create_hidden_state(self, batch_size):  # Output : (num_layers x batch_size x num_outputs)
        h = self.h_init.clone().repeat(1, batch_size, 1)
        c = self.c_init.clone().repeat(1, batch_size, 1)
        return h, c

    def network_size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, inp, prev_states):                                 # inp dimension: (seq_len x batch_size x input_size)
        inp = inp[torch.arange(inp.shape[0]-1, -1, -1), :, :]            # Reversing the input for backward direction
        output, state = self.lstm_network(inp, prev_states)              # Input to LSTM must be of shape (seq_len x batch_size x input_size) in Pytorch. Here, seq_len = 1
        # output = output[torch.arange(output.shape[0]-1, -1, -1), :, :] # Reversing the 'output'.
        return output, state                                             # Output size is (seq_len x batch x hidden_size) as per documentation
