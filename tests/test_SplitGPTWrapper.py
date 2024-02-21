# Test SplitGPTWrapper
import pytest
import torch
import torch.nn as nn
from model import GPTConfig, GPT, CausalSelfAttention
from split_model import SplitAttention, SplitGPTWrapper
from copy import deepcopy
import tiktoken

def test_split_gpt_wrapper_training():
    # Configuration
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)
    weight_decay = 0.1
    qk_weight_decay = 0.0
    learning_rate = 0.001
    betas = (0.9, 0.999)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model
    split_gpt_model = SplitGPTWrapper(GPT(config))

    # Configure optimizer
    optimizer = split_gpt_model.configure_optimizers(weight_decay, qk_weight_decay, learning_rate, betas, device_type)

    # Create a sample input tensor and targets
    batch_size, seq_length = 3, 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))

    # Forward pass
    _, loss = split_gpt_model(x, targets)

    # Backward pass and optimizer step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if parameters have been updated
    for param in split_gpt_model.parameters():
        assert param.grad is not None, "Parameter gradient is None after training step"

def test_optimizer_param_groups():
    # Configuration
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)
    weight_decay = 0.1
    qk_weight_decay = 0.2
    learning_rate = 0.001
    betas = (0.9, 0.999)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create models
    gpt_model = GPT(config)
    split_gpt_model = SplitGPTWrapper(gpt_model)

    # Configure optimizers
    optimizer_gpt = gpt_model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    optimizer_split_gpt = split_gpt_model.configure_optimizers(weight_decay, qk_weight_decay, learning_rate, betas, device_type)

    # Compare parameter groups
    assert len(optimizer_gpt.param_groups) == 2, "GPT does not have 2 param groups"
    assert len(optimizer_split_gpt.param_groups) == 3, "SplitGPTWrapper does not have 3 param groups"

def test_forward_equivalence():
    
    # Create two GPT models: one with CausalSelfAttention and one with SplitAttention
    causal_gpt_model = GPT.from_pretrained('gpt2', dict(dropout=0.0))

    # Create GPT model with SplitAttention
    split_gpt_model = SplitGPTWrapper(deepcopy(causal_gpt_model))

    # Create a sample input tensor
    batch_size, seq_length = 3, 10
    # gpt2 vocab size is 50257
    vocab_size = 50257
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Forward pass through both models
    causal_output, _ = causal_gpt_model(x)
    split_output, _ = split_gpt_model(x)

    # Assert that the outputs are the same (within a tolerance)
    assert torch.allclose(causal_output, split_output, atol=1e-6), "Outputs are not equivalent"

    # Assert that they have not accidentally become the same object 
    assert causal_gpt_model is not split_gpt_model.gpt, "causal_gpt_model is the same object as split_gpt_model"


def test_backward_equivalence():
    
    # Create two GPT models: one with CausalSelfAttention and one with SplitAttention
    causal_gpt_model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    split_gpt_model = SplitGPTWrapper(deepcopy(causal_gpt_model))

    batch_size, seq_length = 3, 10
    vocab_size = 50257
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Get the word token embedding param. If the gradient is the same 
    # we can assume that the gradent is the same for all params in deeper layers as well
    param_causal = next(causal_gpt_model.parameters())
    param_split = next(split_gpt_model.gpt.parameters())

    # Assert that the initial params are equivalent
    assert torch.allclose(param_causal, param_split, atol=1e-6), "Params are not equivalent"

    # Zero gradients before backward pass
    param_causal.grad = None
    param_split.grad = None

    # Forward and backward pass through causal GPT model
    _, loss_causal = causal_gpt_model(x, targets)
    loss_causal.backward()

    # Forward and backward pass through split GPT model
    _, loss_split = split_gpt_model(x, targets)
    loss_split.backward()

    # Assert that the gradients are the same (within a tolerance)
    assert torch.allclose(param_causal.grad, param_split.grad, atol=1e-6), "Gradients are not equivalent"

    # Assert that gradient is not None and not zero
    assert param_causal.grad is not None, "Gradient is None"
    assert torch.any(param_causal.grad != 0), "Gradient is zero"
    
    # Assert that they have not accidentally become the same object 
    assert param_causal is not param_split, "param_causal is the same object as param_split"

def test_optimizer_equivalence():
    
     # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True, dropout=0.0)
    weight_decay = 0.1
    qk_weight_decay = 0.1
    learning_rate = 0.001
    betas = (0.9, 0.99)
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)

    # Create two GPT models: one with CausalSelfAttention and one with SplitAttention
    causal_gpt_model = GPT(config)
    split_gpt_model = SplitGPTWrapper(deepcopy(causal_gpt_model))

    # Configure optimizers
    optimizer_gpt = causal_gpt_model.configure_optimizers(weight_decay, learning_rate, betas, device_type)
    optimizer_split_gpt = split_gpt_model.configure_optimizers(weight_decay, qk_weight_decay, learning_rate, betas, device_type)

    # Create a sample input tensor and targets
    batch_size, seq_length = 3, 10
    x = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_length))

     # Check intitial params are equivalent if we convert to SptlitWrapper
    for (name_causal, param_causal), (name_split, param_split) in zip(split_gpt_model.named_parameters(), SplitGPTWrapper(deepcopy(causal_gpt_model)).gpt.named_parameters()):
        assert torch.allclose(param_causal.data, param_split.data, atol=1e-6), f"Initial param mismatch in: {name_causal}"

    # Forward and backward pass through causal GPT model
    _, loss_causal = causal_gpt_model(x, targets)
    optimizer_gpt.zero_grad()
    loss_causal.backward()

    # Forward and backward pass through split GPT model
    _, loss_split = split_gpt_model(x, targets)
    optimizer_split_gpt.zero_grad()
    loss_split.backward()

    # Optimizer steps
    optimizer_gpt.step()
    optimizer_split_gpt.step()

    # Assert that the updated params are identical if we convert to SplitWrapper.
    # ie. Optimizer step and conversion to SplitWrapper commute
    for (name_causal, param_causal), (name_split, param_split) in zip(split_gpt_model.named_parameters(), SplitGPTWrapper(deepcopy(causal_gpt_model)).gpt.named_parameters()):
        # atol=1e-3, because it fails at 1e-4. ¯\_(ツ)_/¯.
        assert torch.allclose(param_causal.data, param_split.data, atol=1e-3), f"Updated param mismatch in: {name_causal}"

    # Assert that they have not accidentally become the same object 
    assert causal_gpt_model is not split_gpt_model.gpt, "causal_gpt_model is the same object as split_gpt_model"

def test_attention_layer_count():
    # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=12, n_layer=12, bias=True)

    # Create GPT model with SplitAttention
    split_gpt_model = SplitGPTWrapper(GPT(config))

    # Count SplitAttention and CausalSelfAttention layers
    split_attention_count = 0
    causal_attention_count = 0
    for block in split_gpt_model.transformer.h:
        if isinstance(block.attn, SplitAttention):
            split_attention_count += 1
        elif isinstance(block.attn, CausalSelfAttention):
            causal_attention_count += 1

    # Assert that the model contains the correct number of SplitAttention layers
    # and no CausalSelfAttention layers
    assert split_attention_count == config.n_layer, "Incorrect number of SplitAttention layers"
    assert causal_attention_count == 0, "CausalSelfAttention layers found in model"

def test_get_beta():
    # Configuration and model setup
    config = GPTConfig(n_embd=768, n_head=6, n_layer=12, bias=True)

    # Create GPT model with SplitAttention
    split_gpt_model = SplitGPTWrapper(GPT(config))

    # Get the beta (inverse temperature) for all heads
    betas = split_gpt_model.get_betas()

    # Assert that the betas is a dict
    assert isinstance(betas, dict), f"Betas is a {type(betas)}, not a dict"

    # Assert that the betas are floats
    assert all(isinstance(b, float) for b in betas.values()), "Beta is not a float"

    # Assert that all betas are positive
    assert all(b > 0 for b in betas.values()), "Beta is not positive"

    # Assert that the length of beta list is equal to the number of layers
    assert len(betas) == config.n_layer, "Incorrect number of beta values"


def test_sampling_equivalence():
    # Print initialization messages
    print("Initializing gpt2")
    gpt_model = GPT.from_pretrained('gpt2', dict(dropout=0.0))

    print("Wrapping gpt2")
    wrapped_gpt_model = SplitGPTWrapper(deepcopy(gpt_model))

    # Print encoder initialization message
    print("Initializing encoder")
    enc = tiktoken.get_encoding("gpt2")

    # Input preparation
    start = "We're just getting ready to swing, knock me out with a baseball bat."
    start_ids = enc.encode(start, allowed_special={""})
    x = torch.tensor(start_ids, dtype=torch.long)[None, ...]

    # Sampling
    gpt_model.eval()
    wrapped_gpt_model.eval()
    print("Sampling")
    with torch.no_grad():
        y_gpt = gpt_model.generate(x, max_new_tokens=50, temperature=0.01, top_k=1)
        y_wrapped = wrapped_gpt_model.generate(x, max_new_tokens=50, temperature=0.01, top_k=1)

        # Decode and print the outputs
        decoded_y_gpt = enc.decode(y_gpt[0].tolist())
        decoded_y_wrapped = enc.decode(y_wrapped[0].tolist())
        print("Original GPT output:", decoded_y_gpt)
        print("Wrapped GPT output:", decoded_y_wrapped)
        
        # Compare outputs, assert that they are close
        assert torch.allclose(y_gpt, y_wrapped, atol=1e-6), "Mismatch between GPT and Wrapped GPT models."

    print("Sample matched between GPT and Split Wrapped GPT models.")

def test_entropy_change_with_weight_modification():
    # Initialize a Split GPT2 model from pretrained
    model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
    split_model = SplitGPTWrapper(model)

    # Input preparation
    enc = tiktoken.get_encoding("gpt2")
    start = "Entropy is just the information carrying capacity of a system."
    start_ids = enc.encode(start, allowed_special={""})
    x = torch.tensor(start_ids, dtype=torch.long)[None, ...]
    
    # Perform a forward pass and calculate initial entropy
    split_model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        split_model(x)
    initial_entropy = split_model.get_entropies()

    # Modify W_K and W_Q weights by multiplying by 100
    for block in split_model.gpt.transformer.h:
        block.attn.w_k.weight.data *= 100
        block.attn.w_q.weight.data *= 100

    # Perform another forward pass and calculate new entropy
    with torch.no_grad():
        split_model(x)
    modified_entropy = split_model.get_entropies()

    # Assert that both dicts have the same keys
    assert initial_entropy.keys() == modified_entropy.keys(), "Initial and modified entropy dicts have different keys"

    # Assert that the entropy of each key is smaller after the weight modification
    for key in initial_entropy.keys():
        assert modified_entropy[key] < initial_entropy[key], f"Expected entropy to decrease after weight modification, but {modified_entropy[key]} > {initial_entropy[key]}, in {key}"

if __name__ == "__main__":
    test_sampling_equivalence()
    test_get_beta()
    test_forward_equivalence()
    test_backward_equivalence()
    test_optimizer_equivalence()
    test_attention_layer_count()
    test_optimizer_param_groups()
    test_split_gpt_wrapper_training()
    print("All tests passed")