import numpy as np

def softmax(x, axis=-1):
    # x: (..., n) → same shape out
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))   # (..., n) - scalar → (..., n)
    sum_exp = np.sum(exp, axis=axis, keepdims=True)          # (..., n) → (..., 1)
    return exp / sum_exp                                     # (..., n) / (..., 1) → (..., n)

def softmax_backward(softmax_output, d_out):
    # d_input_j = s_j * (d_out_j - sum_k(s_k * d_out_k))
    dot = np.sum(softmax_output * d_out, axis=-1, keepdims=True)  # (..., 1)
    return softmax_output * (d_out - dot)                          # (..., n)

def gelu(x):
    sqrt_2_over_pi = np.sqrt(2 / np.pi)                     # scalar
    inner = sqrt_2_over_pi * (x + 0.044715 * x**3)          # (...) * (...) → (...)
    return 0.5 * x * (1 + np.tanh(inner))                   # (...) * (...) → (...)

def gelu_derivative(x):
    sqrt_2_over_pi = np.sqrt(2 / np.pi)                     # scalar
    inner = sqrt_2_over_pi * (x + 0.044715 * x**3)          # (...) * (...) → (...)
    tanh_inner = np.tanh(inner)                               # (...) → (...)
    sech2_inner = 1 - tanh_inner**2                          # (...) → (...)
    return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2_inner * sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2)  # (...) + (...) → (...)