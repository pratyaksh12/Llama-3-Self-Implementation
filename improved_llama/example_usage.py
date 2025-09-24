import torch
from pathlib import Path

# Import from our improved module
from model import (
    ModelArguments,
    TransformerModel,
)
from tokenizer import (
    LlamaTokenizer,
    ChatMessageFormatter,
    ChatMessage,
)


def create_example_model_configuration():
    """Create example model configuration with verbose parameter names"""
    model_configuration = ModelArguments(
        embedding_dimension=512,  # Smaller for example
        number_of_layers=8,
        number_of_attention_heads=8,
        number_of_key_value_heads=4,  # For grouped query attention
        vocabulary_size=32000,
        multiple_of=256,
        feedforward_dimension_multiplier=None,
        normalization_epsilon=1e-5,
        rope_theta=500000,
        maximum_batch_size=4,
        maximum_sequence_length=512,
    )
    return model_configuration


def demonstrate_model_initialization():
    """Demonstrate how to initialize the improved transformer model"""
    print("Creating model configuration...")
    model_configuration = create_example_model_configuration()
    
    print("Initializing transformer model...")
    transformer_model = TransformerModel(model_configuration)
    
    print(f"Model initialized with {transformer_model.number_of_layers} layers")
    print(f"Embedding dimension: {model_configuration.embedding_dimension}")
    print(f"Vocabulary size: {model_configuration.vocabulary_size}")
    
    return transformer_model


def demonstrate_tokenizer_usage():
    """Demonstrate tokenizer usage (requires actual model file)"""
    # Note: This would require an actual tokenizer model file
    # For demonstration purposes, we'll show the API usage
    
    print("\nTokenizer API demonstration:")
    print("# To initialize tokenizer:")
    print("# tokenizer = LlamaTokenizer('path/to/tokenizer.model')")
    print("# formatter = ChatMessageFormatter(tokenizer)")
    
    print("\n# Example chat message:")
    example_message: ChatMessage = {
        "role": "user",
        "content": "Hello, how are you?"
    }
    print(f"Message: {example_message}")
    
    print("\n# To encode:")
    print("# tokens = formatter.encode_complete_message(example_message)")
    print("# text = tokenizer.decode_tokens_to_text(tokens)")


def demonstrate_forward_pass():
    """Demonstrate a forward pass with dummy data"""
    print("\nDemonstrating forward pass...")
    
    # Create model
    model_configuration = create_example_model_configuration()
    transformer_model = TransformerModel(model_configuration)
    
    # Create dummy input tokens
    batch_size = 2
    sequence_length = 10
    dummy_input_tokens = torch.randint(
        0, 
        model_configuration.vocabulary_size, 
        (batch_size, sequence_length)
    )
    
    print(f"Input shape: {dummy_input_tokens.shape}")
    
    # Forward pass
    with torch.no_grad():
        model_output = transformer_model(dummy_input_tokens, starting_position=0)
    
    print(f"Output shape: {model_output.shape}")
    print(f"Output vocabulary logits: {model_output.shape[-1]}")


if __name__ == "__main__":
    print("=== Improved Llama Model Example ===")
    print("This example demonstrates the refactored model with verbose variable names.\n")
    
    # Demonstrate model initialization
    model = demonstrate_model_initialization()
    
    # Demonstrate tokenizer API
    demonstrate_tokenizer_usage()
    
    # Demonstrate forward pass
    demonstrate_forward_pass()
    
    print("\n=== Example completed successfully! ===")
    print("Key improvements:")
    print("- Verbose variable names (e.g., 'embedding_dimension' instead of 'dim')")
    print("- PyTorch-only dependencies (removed fairscale)")
    print("- No inline list comprehensions")
    print("- Single-line comments for all functions")
    print("- Clear class and function documentation")
