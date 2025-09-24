# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as torch_functional
from torch import nn


@dataclass
class ModelArguments:
    """Configuration class for model hyperparameters"""
    embedding_dimension: int = 4096
    number_of_layers: int = 32
    number_of_attention_heads: int = 32
    number_of_key_value_heads: Optional[int] = None
    vocabulary_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    feedforward_dimension_multiplier: Optional[float] = None
    normalization_epsilon: float = 1e-5
    rope_theta: float = 500000

    maximum_batch_size: int = 32
    maximum_sequence_length: int = 2048


class RootMeanSquareNormalization(torch.nn.Module):
    """Root Mean Square Layer Normalization implementation"""
    
    def __init__(self, embedding_dimension: int, epsilon: float = 1e-6):
        # Initialize RMS normalization layer with dimension and epsilon
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(embedding_dimension))

    def _normalize_tensor(self, input_tensor):
        # Apply RMS normalization to input tensor
        return input_tensor * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, input_tensor):
        # Forward pass through RMS normalization layer
        normalized_output = self._normalize_tensor(input_tensor.float()).type_as(input_tensor)
        return normalized_output * self.weight


def precompute_rotary_frequencies(embedding_dimension: int, sequence_end: int, theta: float = 10000.0):
    """Precompute rotary position embedding frequencies for efficient attention computation"""
    # Create frequency tensor for rotary embeddings
    frequency_range = torch.arange(0, embedding_dimension, 2)
    frequency_slice = frequency_range[: (embedding_dimension // 2)].float()
    frequencies = 1.0 / (theta ** (frequency_slice / embedding_dimension))
    
    # Generate time steps and compute outer product
    time_steps = torch.arange(sequence_end, device=frequencies.device, dtype=torch.float32)
    frequency_matrix = torch.outer(time_steps, frequencies)
    
    # Convert to complex exponential form
    complex_frequencies = torch.polar(torch.ones_like(frequency_matrix), frequency_matrix)
    return complex_frequencies


def reshape_frequencies_for_broadcasting(complex_frequencies: torch.Tensor, input_tensor: torch.Tensor):
    """Reshape frequency tensor to match input tensor dimensions for broadcasting"""
    tensor_dimensions = input_tensor.ndim
    assert 0 <= 1 < tensor_dimensions
    assert complex_frequencies.shape == (input_tensor.shape[1], input_tensor.shape[-1])
    
    # Create shape list for broadcasting
    broadcasting_shape = []
    for dimension_index, dimension_size in enumerate(input_tensor.shape):
        if dimension_index == 1 or dimension_index == tensor_dimensions - 1:
            broadcasting_shape.append(dimension_size)
        else:
            broadcasting_shape.append(1)
    
    return complex_frequencies.view(*broadcasting_shape)


def apply_rotary_position_embedding(
    query_tensor: torch.Tensor,
    key_tensor: torch.Tensor,
    complex_frequencies: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors"""
    # Convert query and key tensors to complex representation
    query_complex = torch.view_as_complex(query_tensor.float().reshape(*query_tensor.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key_tensor.float().reshape(*key_tensor.shape[:-1], -1, 2))
    
    # Reshape frequencies for broadcasting
    reshaped_frequencies = reshape_frequencies_for_broadcasting(complex_frequencies, query_complex)
    
    # Apply rotary embeddings
    rotated_query = torch.view_as_real(query_complex * reshaped_frequencies).flatten(3)
    rotated_key = torch.view_as_real(key_complex * reshaped_frequencies).flatten(3)
    
    return rotated_query.type_as(query_tensor), rotated_key.type_as(key_tensor)


def repeat_key_value_heads(input_tensor: torch.Tensor, repetition_count: int) -> torch.Tensor:
    """Repeat key-value heads to match number of query heads for grouped query attention"""
    batch_size, sequence_length, number_kv_heads, head_dimension = input_tensor.shape
    
    if repetition_count == 1:
        return input_tensor
    
    # Expand tensor dimensions for repetition
    expanded_tensor = input_tensor[:, :, :, None, :]
    repeated_tensor = expanded_tensor.expand(batch_size, sequence_length, number_kv_heads, repetition_count, head_dimension)
    reshaped_tensor = repeated_tensor.reshape(batch_size, sequence_length, number_kv_heads * repetition_count, head_dimension)
    
    return reshaped_tensor


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with key-value caching"""
    
    def __init__(self, model_arguments: ModelArguments):
        # Initialize multi-head attention layer
        super().__init__()
        self.number_of_key_value_heads = model_arguments.number_of_attention_heads if model_arguments.number_of_key_value_heads is None else model_arguments.number_of_key_value_heads
        self.number_of_local_heads = model_arguments.number_of_attention_heads
        self.number_of_local_kv_heads = self.number_of_key_value_heads
        self.repetition_factor = self.number_of_local_heads // self.number_of_local_kv_heads
        self.head_dimension = model_arguments.embedding_dimension // model_arguments.number_of_attention_heads

        # Query projection layer
        self.query_projection = nn.Linear(
            model_arguments.embedding_dimension,
            model_arguments.number_of_attention_heads * self.head_dimension,
            bias=False
        )
        
        # Key projection layer
        self.key_projection = nn.Linear(
            model_arguments.embedding_dimension,
            self.number_of_key_value_heads * self.head_dimension,
            bias=False
        )
        
        # Value projection layer
        self.value_projection = nn.Linear(
            model_arguments.embedding_dimension,
            self.number_of_key_value_heads * self.head_dimension,
            bias=False
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(
            model_arguments.number_of_attention_heads * self.head_dimension,
            model_arguments.embedding_dimension,
            bias=False
        )

        # Initialize key-value cache tensors (device will be set during forward pass)
        cache_shape = (
            model_arguments.maximum_batch_size,
            model_arguments.maximum_sequence_length,
            self.number_of_local_kv_heads,
            self.head_dimension,
        )
        self.key_cache = torch.zeros(cache_shape)
        self.value_cache = torch.zeros(cache_shape)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        starting_position: int,
        rotary_frequencies: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        # Forward pass through multi-head attention mechanism
        batch_size, sequence_length, _ = input_embeddings.shape
        
        # Project input to query, key, and value
        query_states = self.query_projection(input_embeddings)
        key_states = self.key_projection(input_embeddings)
        value_states = self.value_projection(input_embeddings)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, sequence_length, self.number_of_local_heads, self.head_dimension)
        key_states = key_states.view(batch_size, sequence_length, self.number_of_local_kv_heads, self.head_dimension)
        value_states = value_states.view(batch_size, sequence_length, self.number_of_local_kv_heads, self.head_dimension)

        # Apply rotary position embeddings
        query_with_rope, key_with_rope = apply_rotary_position_embedding(query_states, key_states, complex_frequencies=rotary_frequencies)

        # Update cache with current key-value states
        self.key_cache = self.key_cache.to(query_with_rope)
        self.value_cache = self.value_cache.to(query_with_rope)

        # Store current keys and values in cache
        end_position = starting_position + sequence_length
        self.key_cache[:batch_size, starting_position:end_position] = key_with_rope
        self.value_cache[:batch_size, starting_position:end_position] = value_states

        # Retrieve keys and values from cache
        cached_keys = self.key_cache[:batch_size, :end_position]
        cached_values = self.value_cache[:batch_size, :end_position]

        # Repeat key-value heads if needed for grouped query attention
        repeated_keys = repeat_key_value_heads(cached_keys, self.repetition_factor)
        repeated_values = repeat_key_value_heads(cached_values, self.repetition_factor)

        # Transpose for attention computation
        query_transposed = query_with_rope.transpose(1, 2)
        keys_transposed = repeated_keys.transpose(1, 2)
        values_transposed = repeated_values.transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_transposed, keys_transposed.transpose(2, 3)) / math.sqrt(self.head_dimension)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        # Apply softmax to get attention probabilities
        attention_probabilities = torch_functional.softmax(attention_scores.float(), dim=-1).type_as(query_with_rope)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probabilities, values_transposed)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        final_output = self.output_projection(attention_output)
        
        return final_output


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with SwiGLU activation"""
    
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        multiple_of: int,
        feedforward_dimension_multiplier: Optional[float],
    ):
        # Initialize feed-forward network layers
        super().__init__()
        hidden_dimension = int(2 * hidden_dimension / 3)
        
        # Apply custom dimension factor multiplier if provided
        if feedforward_dimension_multiplier is not None:
            hidden_dimension = int(feedforward_dimension_multiplier * hidden_dimension)
            
        # Round up to nearest multiple for efficiency
        hidden_dimension = multiple_of * ((hidden_dimension + multiple_of - 1) // multiple_of)

        self.gate_projection = nn.Linear(input_dimension, hidden_dimension, bias=False)
        self.down_projection = nn.Linear(hidden_dimension, input_dimension, bias=False)
        self.up_projection = nn.Linear(input_dimension, hidden_dimension, bias=False)

    def forward(self, input_tensor):
        # Forward pass through SwiGLU feed-forward network
        gate_output = torch_functional.silu(self.gate_projection(input_tensor))
        up_output = self.up_projection(input_tensor)
        combined_output = gate_output * up_output
        final_output = self.down_projection(combined_output)
        return final_output


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward layers"""
    
    def __init__(self, layer_identifier: int, model_arguments: ModelArguments):
        # Initialize transformer block with attention and feed-forward layers
        super().__init__()
        self.number_of_heads = model_arguments.number_of_attention_heads
        self.embedding_dimension = model_arguments.embedding_dimension
        self.head_dimension = model_arguments.embedding_dimension // model_arguments.number_of_attention_heads
        
        self.multi_head_attention = MultiHeadAttention(model_arguments)
        self.feed_forward_network = FeedForwardNetwork(
            input_dimension=model_arguments.embedding_dimension,
            hidden_dimension=4 * model_arguments.embedding_dimension,
            multiple_of=model_arguments.multiple_of,
            feedforward_dimension_multiplier=model_arguments.feedforward_dimension_multiplier,
        )
        
        self.layer_identifier = layer_identifier
        self.attention_normalization = RootMeanSquareNormalization(model_arguments.embedding_dimension, epsilon=model_arguments.normalization_epsilon)
        self.feedforward_normalization = RootMeanSquareNormalization(model_arguments.embedding_dimension, epsilon=model_arguments.normalization_epsilon)

    def forward(
        self,
        input_embeddings: torch.Tensor,
        starting_position: int,
        rotary_frequencies: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        # Forward pass through transformer block with residual connections
        # Apply attention with residual connection
        normalized_input = self.attention_normalization(input_embeddings)
        attention_output = self.multi_head_attention(normalized_input, starting_position, rotary_frequencies, attention_mask)
        hidden_states = input_embeddings + attention_output
        
        # Apply feed-forward with residual connection
        normalized_hidden = self.feedforward_normalization(hidden_states)
        feedforward_output = self.feed_forward_network(normalized_hidden)
        final_output = hidden_states + feedforward_output
        
        return final_output


class TransformerModel(nn.Module):
    """Complete transformer model with embedding, layers, and output projection"""
    
    def __init__(self, model_parameters: ModelArguments):
        # Initialize complete transformer model
        super().__init__()
        self.model_parameters = model_parameters
        self.vocabulary_size = model_parameters.vocabulary_size
        self.number_of_layers = model_parameters.number_of_layers

        # Token embedding layer
        self.token_embeddings = nn.Embedding(model_parameters.vocabulary_size, model_parameters.embedding_dimension)

        # Create transformer layers
        self.transformer_layers = torch.nn.ModuleList()
        for layer_index in range(model_parameters.number_of_layers):
            transformer_block = TransformerBlock(layer_index, model_parameters)
            self.transformer_layers.append(transformer_block)

        # Final normalization and output projection
        self.final_normalization = RootMeanSquareNormalization(model_parameters.embedding_dimension, epsilon=model_parameters.normalization_epsilon)
        self.output_projection = nn.Linear(model_parameters.embedding_dimension, model_parameters.vocabulary_size, bias=False)

        # Precompute rotary frequencies
        head_dimension = model_parameters.embedding_dimension // model_parameters.number_of_attention_heads
        max_position = model_parameters.maximum_sequence_length * 2
        self.rotary_frequencies = precompute_rotary_frequencies(
            head_dimension,
            max_position,
            model_parameters.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, input_tokens: torch.Tensor, starting_position: int):
        # Forward pass through complete transformer model
        _batch_size, sequence_length = input_tokens.shape
        
        # Get token embeddings
        hidden_states = self.token_embeddings(input_tokens)
        
        # Move rotary frequencies to same device as input
        self.rotary_frequencies = self.rotary_frequencies.to(hidden_states.device)
        end_position = starting_position + sequence_length
        current_frequencies = self.rotary_frequencies[starting_position:end_position]

        # Create causal attention mask if needed
        attention_mask = None
        if sequence_length > 1:
            # Create lower triangular mask for causal attention
            mask_shape = (sequence_length, sequence_length)
            causal_mask = torch.full(mask_shape, float("-inf"), device=input_tokens.device)
            causal_mask = torch.triu(causal_mask, diagonal=1)

            # Concatenate with zeros for past positions in cache
            past_positions_mask = torch.zeros((sequence_length, starting_position), device=input_tokens.device)
            attention_mask = torch.hstack([past_positions_mask, causal_mask]).type_as(hidden_states)

        # Pass through all transformer layers
        for transformer_layer in self.transformer_layers:
            hidden_states = transformer_layer(hidden_states, starting_position, current_frequencies, attention_mask)
            
        # Apply final normalization and output projection
        normalized_output = self.final_normalization(hidden_states)
        logits = self.output_projection(normalized_output).float()
        
        return logits
