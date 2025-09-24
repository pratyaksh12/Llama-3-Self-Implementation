import os
from logging import getLogger
from pathlib import Path
from typing import (
    AbstractSet,
    cast,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Sequence,
    TypedDict,
    Union,
)

import tiktoken
from tiktoken.load import load_tiktoken_bpe


logger = getLogger(__name__)


ConversationRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    """Type definition for a single chat message"""
    role: ConversationRole
    content: str


ConversationDialog = Sequence[ChatMessage]


class LlamaTokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer for Llama models.
    """

    special_token_mappings: Dict[str, int]
    number_of_reserved_special_tokens = 256
    pattern_string = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_file_path: str):
        # Initialize the tokenizer with a Tiktoken model file
        assert os.path.isfile(model_file_path), model_file_path

        mergeable_token_ranks = load_tiktoken_bpe(model_file_path)
        number_of_base_tokens = len(mergeable_token_ranks)
        
        # Define core special tokens
        core_special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ]
        
        # Generate additional reserved special tokens
        additional_reserved_tokens = []
        for token_index in range(5, self.number_of_reserved_special_tokens - 5):
            reserved_token = f"<|reserved_special_token_{token_index}|>"
            additional_reserved_tokens.append(reserved_token)
        
        # Combine all special tokens
        all_special_tokens = core_special_tokens + additional_reserved_tokens
        
        # Create mapping from special tokens to their IDs
        self.special_token_mappings = {}
        for token_index, special_token in enumerate(all_special_tokens):
            token_id = number_of_base_tokens + token_index
            self.special_token_mappings[special_token] = token_id
            
        # Initialize the tiktoken encoding model
        self.tiktoken_model = tiktoken.Encoding(
            name=Path(model_file_path).name,
            pat_str=self.pattern_string,
            mergeable_ranks=mergeable_token_ranks,
            special_tokens=self.special_token_mappings,
        )
        logger.info(f"Reloaded tiktoken model from {model_file_path}")

        # Set vocabulary size and special token IDs
        self.vocabulary_size: int = self.tiktoken_model.n_vocab
        self.beginning_of_sequence_token_id: int = self.special_token_mappings["<|begin_of_text|>"]
        self.end_of_sequence_token_id: int = self.special_token_mappings["<|end_of_text|>"]
        self.padding_token_id: int = -1
        
        # Define stop tokens for generation
        stop_token_set = set()
        stop_token_set.add(self.special_token_mappings["<|end_of_text|>"])
        stop_token_set.add(self.special_token_mappings["<|eot_id|>"])
        self.stop_token_set = stop_token_set
        
        logger.info(
            f"#words: {self.vocabulary_size} - BOS ID: {self.beginning_of_sequence_token_id} - EOS ID: {self.end_of_sequence_token_id}"
        )

    def encode_text_to_tokens(
        self,
        input_string: str,
        *,
        add_beginning_of_sequence: bool,
        add_end_of_sequence: bool,
        allowed_special_tokens: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special_tokens: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        # Encode a string into a list of token IDs
        assert type(input_string) is str

        # Constants for tiktoken limits
        TIKTOKEN_MAXIMUM_ENCODE_CHARACTERS = 400_000
        MAXIMUM_CONSECUTIVE_NON_WHITESPACE_CHARACTERS = 25_000

        # Split input string into manageable chunks
        string_chunks = self._generate_string_chunks(
            input_string, 
            TIKTOKEN_MAXIMUM_ENCODE_CHARACTERS, 
            MAXIMUM_CONSECUTIVE_NON_WHITESPACE_CHARACTERS
        )
        
        # Encode each chunk and combine results
        encoded_token_list: List[int] = []
        for string_chunk in string_chunks:
            chunk_tokens = self.tiktoken_model.encode(
                string_chunk,
                allowed_special=allowed_special_tokens,
                disallowed_special=disallowed_special_tokens,
            )
            encoded_token_list.extend(chunk_tokens)
            
        # Add special tokens if requested
        if add_beginning_of_sequence:
            encoded_token_list.insert(0, self.beginning_of_sequence_token_id)
        if add_end_of_sequence:
            encoded_token_list.append(self.end_of_sequence_token_id)
            
        return encoded_token_list

    def decode_tokens_to_text(self, token_sequence: Sequence[int]) -> str:
        # Decode a list of token IDs into a string
        # Typecast is safe here as tiktoken doesn't perform list-specific operations
        token_list = cast(List[int], token_sequence)
        decoded_string = self.tiktoken_model.decode(token_list)
        return decoded_string

    def _generate_string_chunks(self, input_string: str, maximum_chunk_size: int, maximum_consecutive_slice_length: int) -> Iterator[str]:
        # Generate string chunks for processing within tiktoken limits
        string_chunks = []
        
        # Split string into chunks of maximum size
        for chunk_start_index in range(0, len(input_string), maximum_chunk_size):
            chunk_end_index = chunk_start_index + maximum_chunk_size
            current_chunk = input_string[chunk_start_index:chunk_end_index]
            
            # Further split chunk by whitespace/non-whitespace patterns
            whitespace_split_chunks = self._split_by_whitespace_patterns(
                current_chunk, 
                maximum_consecutive_slice_length
            )
            
            string_chunks.extend(whitespace_split_chunks)
            
        return iter(string_chunks)

    @staticmethod
    def _split_by_whitespace_patterns(input_string: str, maximum_consecutive_slice_length: int) -> Iterator[str]:
        # Split string so each substring has limited consecutive whitespace or non-whitespace characters
        current_slice_length = 0
        current_slice_start_index = 0
        
        if len(input_string) == 0:
            return iter([])
            
        current_slice_is_whitespace = input_string[0].isspace()

        string_chunks = []
        for character_index in range(len(input_string)):
            character_is_whitespace = input_string[character_index].isspace()

            # Check if we're switching between whitespace and non-whitespace
            if current_slice_is_whitespace != character_is_whitespace:
                current_slice_length = 1
                current_slice_is_whitespace = character_is_whitespace
            else:
                current_slice_length += 1
                
                # Split if we exceed the maximum consecutive length
                if current_slice_length > maximum_consecutive_slice_length:
                    chunk = input_string[current_slice_start_index:character_index]
                    string_chunks.append(chunk)
                    current_slice_start_index = character_index
                    current_slice_length = 1
                    
        # Add the final chunk
        final_chunk = input_string[current_slice_start_index:]
        string_chunks.append(final_chunk)
        
        return iter(string_chunks)


class ChatMessageFormatter:
    """Formatter for encoding chat conversations in Llama format"""
    
    def __init__(self, tokenizer_instance: LlamaTokenizer):
        # Initialize chat formatter with tokenizer instance
        self.tokenizer_instance = tokenizer_instance

    def encode_message_header(self, chat_message: ChatMessage) -> List[int]:
        # Encode the header portion of a chat message
        header_token_list = []
        
        # Add start header token
        start_header_token_id = self.tokenizer_instance.special_token_mappings["<|start_header_id|>"]
        header_token_list.append(start_header_token_id)
        
        # Encode the role
        role_tokens = self.tokenizer_instance.encode_text_to_tokens(
            chat_message["role"], 
            add_beginning_of_sequence=False, 
            add_end_of_sequence=False
        )
        header_token_list.extend(role_tokens)
        
        # Add end header token
        end_header_token_id = self.tokenizer_instance.special_token_mappings["<|end_header_id|>"]
        header_token_list.append(end_header_token_id)
        
        # Add newlines
        newline_tokens = self.tokenizer_instance.encode_text_to_tokens(
            "\n\n", 
            add_beginning_of_sequence=False, 
            add_end_of_sequence=False
        )
        header_token_list.extend(newline_tokens)
        
        return header_token_list

    def encode_complete_message(self, chat_message: ChatMessage) -> List[int]:
        # Encode a complete chat message including header and content
        message_token_list = self.encode_message_header(chat_message)
        
        # Encode message content
        content_tokens = self.tokenizer_instance.encode_text_to_tokens(
            chat_message["content"].strip(), 
            add_beginning_of_sequence=False, 
            add_end_of_sequence=False
        )
        message_token_list.extend(content_tokens)
        
        # Add end of turn token
        end_of_turn_token_id = self.tokenizer_instance.special_token_mappings["<|eot_id|>"]
        message_token_list.append(end_of_turn_token_id)
        
        return message_token_list

    def encode_conversation_dialog(self, conversation_dialog: ConversationDialog) -> List[int]:
        # Encode an entire conversation dialog for model input
        dialog_token_list = []
        
        # Add beginning of text token
        beginning_of_text_token_id = self.tokenizer_instance.special_token_mappings["<|begin_of_text|>"]
        dialog_token_list.append(beginning_of_text_token_id)
        
        # Encode each message in the dialog
        for chat_message in conversation_dialog:
            message_tokens = self.encode_complete_message(chat_message)
            dialog_token_list.extend(message_tokens)
            
        # Add the start of an assistant message for the model to complete
        assistant_message_start = {"role": "assistant", "content": ""}
        assistant_header_tokens = self.encode_message_header(assistant_message_start)
        dialog_token_list.extend(assistant_header_tokens)
        
        return dialog_token_list
