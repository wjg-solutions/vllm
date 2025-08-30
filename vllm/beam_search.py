# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, Set

from vllm.lora.request import LoRARequest
from vllm.sequence import Logprob

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalDataDict


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """

    # The tokens include the prompt.
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: Optional[LoRARequest] = None
    cum_logprob: float = 0.0
    text: Optional[str] = None
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None
    multi_modal_data: Optional["MultiModalDataDict"] = None
    mm_processor_kwargs: Optional[dict[str, Any]] = None
    # EOS handling fields
    is_finished: bool = False
    finished_step: Optional[int] = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """

    sequences: list[BeamSearchSequence]


@dataclass
class EOSTokenConfig:
    """Configuration for EOS token handling in beam search."""

    # Primary EOS token ID (usually from tokenizer.eos_token_id)
    primary_eos_token_id: Optional[int] = None
    # Additional EOS token IDs (e.g., from generation config)
    additional_eos_token_ids: Set[int] = None
    # Whether to ignore EOS tokens and continue generation
    ignore_eos: bool = False
    # Minimum number of tokens before EOS can terminate sequence
    min_tokens: int = 0

    def __post_init__(self):
        if self.additional_eos_token_ids is None:
            self.additional_eos_token_ids = set()

    @property
    def all_eos_token_ids(self) -> Set[int]:
        """Get all EOS token IDs (primary + additional)."""
        eos_ids = set(self.additional_eos_token_ids)
        if self.primary_eos_token_id is not None:
            eos_ids.add(self.primary_eos_token_id)
        return eos_ids

    def is_eos_token(self, token_id: int) -> bool:
        """Check if a token ID is an EOS token."""
        return token_id in self.all_eos_token_ids

    def should_stop_at_eos(self, tokens: list[int], current_step: int) -> bool:
        """Determine if sequence should stop at EOS token."""
        if self.ignore_eos:
            return False

        # Check minimum token requirement
        if len(tokens) < self.min_tokens:
            return False

        return True


class BeamSearchInstance:
    def __init__(
        self,
        prompt_tokens: list[int],
        lora_request: Optional[LoRARequest] = None,
        logprobs: Optional[list[dict[int, Logprob]]] = None,
        eos_config: Optional[EOSTokenConfig] = None,
        **kwargs,
    ):
        self.beams: list[BeamSearchSequence] = [
            BeamSearchSequence(
                tokens=prompt_tokens,
                logprobs=[] if logprobs is None else list(logprobs),
                lora_request=lora_request,
                **kwargs,
            )
        ]
        self.completed: list[BeamSearchSequence] = []
        self.eos_config = eos_config or EOSTokenConfig()
        self.current_step = 0

    def should_terminate_beam(
        self, beam: BeamSearchSequence, new_token_id: int
    ) -> bool:
        """Check if a beam should be terminated due to EOS token."""
        if not self.eos_config.is_eos_token(new_token_id):
            return False

        new_tokens = beam.tokens + [new_token_id]
        return self.eos_config.should_stop_at_eos(new_tokens, self.current_step)

    def finalize_beam(
        self, beam: BeamSearchSequence, finish_reason: str = "stop"
    ) -> None:
        """Mark a beam as finished and set completion metadata."""
        beam.is_finished = True
        beam.finished_step = self.current_step
        beam.finish_reason = finish_reason
        if self.eos_config.is_eos_token(beam.tokens[-1]):
            beam.stop_reason = beam.tokens[-1]

    def add_completed_beam(self, beam: BeamSearchSequence) -> None:
        """Add a beam to the completed list with proper finalization."""
        if not beam.is_finished:
            self.finalize_beam(beam)
        self.completed.append(beam)

    def step(self) -> None:
        """Advance to the next step in beam search."""
        self.current_step += 1

    def cleanup_finished_beams(self) -> None:
        """Remove finished beams from active beam list to free memory."""
        self.beams = [beam for beam in self.beams if not beam.is_finished]

    def get_best_completed_beams(
        self, beam_width: int, length_penalty: float = 1.0
    ) -> list[BeamSearchSequence]:
        """Get the best completed beams sorted by score."""
        if not self.completed:
            return []

        # Sort by beam search score
        sorted_completed = sorted(
            self.completed,
            key=lambda x: get_beam_search_score(
                x.tokens,
                x.cum_logprob,
                self.eos_config.primary_eos_token_id or 0,
                length_penalty,
            ),
            reverse=True,
        )

        return sorted_completed[:beam_width]


def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float:
    """Calculate the beam search score without length penalty.

    Returns the raw cumulative log probability for fair comparison
    between beams of different lengths.
    """
    # Simply return the cumulative log probability without any length normalization
    # This allows beams to be ranked purely by their likelihood
    return cumulative_logprob


def create_sort_beams_key_function(eos_token_id: int, length_penalty: float):
    def sort_beams_key(x: BeamSearchSequence) -> float:
        return get_beam_search_score(
            x.tokens, x.cum_logprob, eos_token_id, length_penalty
        )

    return sort_beams_key


def detect_eos_tokens_from_tokenizer(tokenizer) -> EOSTokenConfig:
    """Detect EOS tokens from tokenizer and create configuration."""
    config = EOSTokenConfig()

    # Get primary EOS token
    if (
        hasattr(tokenizer, "eos_token_id")
        and tokenizer.eos_token_id is not None
    ):
        config.primary_eos_token_id = tokenizer.eos_token_id

    # Get additional EOS tokens from various sources
    additional_eos = set()

    # Check for pad token that might also serve as EOS
    if (
        hasattr(tokenizer, "pad_token_id")
        and tokenizer.pad_token_id is not None
    ):
        # Some models use pad token as EOS
        if (
            hasattr(tokenizer, "eos_token_id")
            and tokenizer.pad_token_id == tokenizer.eos_token_id
        ):
            additional_eos.add(tokenizer.pad_token_id)

    # Check for special tokens that might be EOS variants
    if hasattr(tokenizer, "special_tokens_map"):
        special_tokens = tokenizer.special_tokens_map
        for token_type, token_value in special_tokens.items():
            if "eos" in token_type.lower() or "end" in token_type.lower():
                if isinstance(token_value, str):
                    token_id = tokenizer.convert_tokens_to_ids(token_value)
                    if token_id is not None:
                        additional_eos.add(token_id)

    # Check for model-specific EOS tokens
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        for token, token_id in vocab.items():
            if any(
                eos_pattern in token.lower()
                for eos_pattern in ["</s>", "<eos>", "<|end|>"]
            ):
                additional_eos.add(token_id)

    config.additional_eos_token_ids = additional_eos
    return config


def detect_eos_tokens_from_generation_config(generation_config) -> Set[int]:
    """Extract EOS token IDs from model generation configuration."""
    eos_token_ids = set()

    if hasattr(generation_config, "eos_token_id"):
        if isinstance(generation_config.eos_token_id, (list, tuple)):
            eos_token_ids.update(generation_config.eos_token_id)
        elif generation_config.eos_token_id is not None:
            eos_token_ids.add(generation_config.eos_token_id)

    return eos_token_ids
