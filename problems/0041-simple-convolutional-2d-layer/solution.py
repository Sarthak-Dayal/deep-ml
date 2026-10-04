import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape # H x W
	kernel_height, kernel_width = kernel.shape # kH x kW

	padded_input = np.pad(
		input_matrix, 
		pad_width=padding, 
		mode="constant", 
		constant_values=0
	)

	windows = np.lib.stride_tricks.sliding_window_view(
		padded_input, (kernel_height, kernel_width)
	)

	windows = windows[::stride, ::stride]
	output_matrix = (windows * kernel).sum(axis=(-2,-1))
    
	return output_matrix
