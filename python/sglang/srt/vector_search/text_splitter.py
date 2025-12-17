from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class TokenChunkConfig:
    chunk_tokens: int = 256
    chunk_overlap_tokens: int = 50

    def validate(self) -> None:
        if self.chunk_tokens <= 0:
            raise ValueError(f"chunk_tokens must be > 0, got {self.chunk_tokens}")
        if self.chunk_overlap_tokens < 0:
            raise ValueError(
                f"chunk_overlap_tokens must be >= 0, got {self.chunk_overlap_tokens}"
            )
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError(
                f"chunk_overlap_tokens must be < chunk_tokens, got {self.chunk_overlap_tokens} >= {self.chunk_tokens}"
            )


@dataclass(frozen=True)
class TokenizerAdapter:
    encode_fn: Callable[[str], list[int]]
    decode_fn: Callable[[list[int]], str]

    def encode(self, text: str) -> list[int]:
        return self.encode_fn(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.decode_fn(token_ids)


def split_text_on_tokens(
    *,
    text: str,
    tokenizer: TokenizerLike,
    config: TokenChunkConfig,
) -> list[str]:
    config.validate()

    input_ids = tokenizer.encode(text)
    splits: list[str] = []
    start_idx = 0
    stride = config.chunk_tokens - config.chunk_overlap_tokens

    while start_idx < len(input_ids):
        end_idx = min(start_idx + config.chunk_tokens, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]
        if not chunk_ids:
            break
        chunk_text = tokenizer.decode(chunk_ids).strip()
        if chunk_text:
            splits.append(chunk_text)
        if end_idx == len(input_ids):
            break
        start_idx += stride

    return splits


def make_hf_tokenizer_adapter(tokenizer) -> TokenizerAdapter:
    """
    Create a TokenizerAdapter from a HuggingFace tokenizer.

    The tokenizer is expected to have `.encode` and `.decode` methods compatible with
    transformers tokenizers.
    """

    def _encode(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def _decode(token_ids: list[int]) -> str:
        return str(
            tokenizer.decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        )

    return TokenizerAdapter(encode_fn=_encode, decode_fn=_decode)

