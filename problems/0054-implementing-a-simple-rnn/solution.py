import torch

def rnn_forward(input_sequence: list, initial_hidden_state: list, Wx: list, Wh: list, b: list) -> torch.Tensor:
    """
    Implements a simple RNN cell forward pass using PyTorch.

    Args:
        input_sequence: List of input vectors for each time step.
        initial_hidden_state: The initial hidden state vector.
        Wx: Weight matrix for input-to-hidden connections.
        Wh: Weight matrix for hidden-to-hidden connections.
        b: Bias vector.

    Returns:
        torch.Tensor: The final hidden state after processing the entire sequence,
                      rounded to four decimal places.
    """
    input_seq = torch.tensor(input_sequence)
    hidden_state = torch.tensor(initial_hidden_state)
    input_weight = torch.tensor(Wx)
    hidden_weight = torch.tensor(Wh)
    bias = torch.tensor(b)

    for input_tensor in input_seq:
        inp_contribution = input_weight @ input_tensor
        hidden_contribution = hidden_weight @ hidden_state
        new_hidden = inp_contribution + hidden_contribution + bias
        hidden_state = torch.tanh(new_hidden)
    
    return torch.round(hidden_state * 10000) / 10000
    