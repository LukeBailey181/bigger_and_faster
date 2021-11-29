import numpy as np
import sys
import torch

def quantize(W, q=8):
    """
    quantize the values of a matrix
    :param W: matrix of unquantized values
    :param q: bit width of quantized value

    :return: (quantized matrix, (shift factor, scale factor))
    """
    # set the range of numbers in each quantization level accordingly
    quant_range = 2 ** q - 1

    # shift matrix so that minimum value is 0
    shifted = W - W.min()

    # scale matrix as needed by max val on q bits / max value in shifted matrix
    shifted_max = shifted.max()

    # exit early to avoid divide by 0 if a 1 value array
    if shifted_max == 0:
        return shifted, (W.min(), 1)

    quantized = np.rint(shifted / shifted.max() * quant_range) 
    
    # hypothetically, would change data type in ### commented line
    # however, currently just "pseudo-quantizing" - not actually changing data type
    ###quantized = quantized.astype(np.uint8)

    # return the quantized matrix and (shift, scale) factors to invert quantization
    return quantized, (W.min(), shifted.max() / quant_range)

def dequantize(W, inversion_args):
    """
    Undo quantization for actual use of values in model
    
    :param W: matrix of quantized values
    :param inversion_args: tuple containing (shift factor, scale factor) as is provided by quantize func

    :return: dequantized matrix
    """

    shift, scale = inversion_args
    # undo scaling by factor given in extra args
    unscaled = W * scale
    # undo shift by factor given in extra args
    dequantized = unscaled + shift
    return dequantized

def get_quantized_state_dict(model):
    """
    gets the quantized state dict for a model
    :param model: torch model to quantize 
    :return: a torch state dict for the same model, now quantized - can be loaded by model.load_state_dict
    """

    read = dict(model.state_dict())
    write = { key: torch.from_numpy(dequantize(*quantize(read[key].numpy()))) for key in read.keys() }

    return write

def quantize_model(model):
    """
    quantize a model in place
    :param model: torch model to quantize 
    """
    model.load_state_dict(get_quantized_state_dict(model))

def test_quantization(q=8):
    def count_bytes(x):
        return sys.getsizeof(x)
    def did_pass(x, thresh=2):
        bytes_original = count_bytes(x.astype(np.float32))
        quantized, args = quantize(x, q)
        bytes_quantized = count_bytes(quantized)
        dequantized = dequantize(quantized, args)
        mean_err = np.mean(np.abs(dequantized-x))  
        accurate = mean_err < thresh
        if not accurate:
            print("FAIL: Mean error %f above threshold %f" % (mean_err, thresh))
            return False
        print("Success!")
        return True
    did_pass(np.array([i for i in range(100)]))
    did_pass(np.array([i*.5 for i in range(100)]))
    did_pass(np.random.uniform(0, 10, size=(100, 100)))
    did_pass(np.random.uniform(5, 10, size=(100, 100)))
    did_pass(np.random.uniform(-5, -10, size=(100, 100)))
    did_pass(np.random.normal(-10, 1, size=(100, 100)))
    did_pass(np.random.uniform(100, 100+255, size=(20, 40)))

def test_model_overwrite():
    model = torch.nn.Linear(9, 1)
    print(f"State dict before: {model.state_dict()}")
    quantize_model(model)
    print(f"State dict after: {model.state_dict()}")

if __name__ == "__main__":
    test_model_overwrite()
