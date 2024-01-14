# Test SplitAttention class
import pytest
import torch
import torch.nn as nn
from model import GPTConfig, CausalSelfAttention
from split_model import SplitAttention

def test_split_attention_forward_equivalence():
    # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)
    causal_attention = CausalSelfAttention(config)
    split_attention = SplitAttention(causal_attention)

    # Create a sample input tensor
    batch_size, seq_length, embd_size = 3, 10, 768  # Adjust as needed
    x = torch.randn(batch_size, seq_length, embd_size)

    # Forward pass through both models
    causal_attention_output = causal_attention(x)
    split_attention_output = split_attention(x)
    
    # Assert that the outputs are the same (within a tolerance)
    assert torch.allclose(causal_attention_output, split_attention_output, atol=1e-6), "Outputs are not equivalent"

    # Assert that they have not accidentally become the same object 
    assert causal_attention is not split_attention, "causal_attention is the same object as split_attention"

def test_split_attention_backward_equivalence():
    # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)
    causal_attention = CausalSelfAttention(config)
    split_attention = SplitAttention(causal_attention)

    # Create a sample input tensor and target
    batch_size, seq_length, embd_size = 3, 10, 768
    x_causal = torch.randn(batch_size, seq_length, embd_size, requires_grad=True)
    x_split = x_causal.clone().detach().requires_grad_(True)  # Clone and detach for separate computation

    # Define a simple loss function and a target for comparison
    target = torch.randn(batch_size, seq_length, embd_size)
    loss_fn = nn.MSELoss()

    # Forward and backward pass through causal_attention
    output_causal = causal_attention(x_causal)
    loss_causal = loss_fn(output_causal, target)
    loss_causal.backward()

    # Forward and backward pass through split_attention
    output_split = split_attention(x_split)
    loss_split = loss_fn(output_split, target)
    loss_split.backward()

    # Assert that the gradients are the same (within a tolerance)
    assert torch.allclose(x_causal.grad, x_split.grad, atol=1e-6), "Gradients are not equivalent"

    # Assert that they have not accidentally become the same object 
    assert causal_attention is not split_attention, "causal_attention is the same object as split_attention"

def test_get_magnitudes():
    # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)
    causal_attention = CausalSelfAttention(config)
    split_attention = SplitAttention(causal_attention)

    # Calculate average magnitude using the method
    calculated_magnitude = split_attention.get_average_magnitude()

    # Manually calculate the expected magnitude for comparison
    # Transpose w_k.weight to match dimensions for matrix multiplication
    W_KT = split_attention.w_k.weight.t()
    W_Q_W_KT = torch.matmul(split_attention.w_q.weight, W_KT)
    expected_magnitude = torch.norm(W_Q_W_KT)

    # Assert that the calculated magnitude matches the manually computed magnitude (within a tolerance)
    assert torch.allclose(calculated_magnitude, expected_magnitude, atol=1e-6), "Calculated magnitude does not match the expected magnitude"


if __name__ == "__main__":
    test_split_attention_forward_equivalence()
    test_split_attention_backward_equivalence()
    test_get_magnitudes()
    print("All tests passed")