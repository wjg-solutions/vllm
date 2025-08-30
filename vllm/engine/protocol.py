# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Iterable, Mapping, Optional, Union

from vllm.beam_search import (BeamSearchSequence, create_sort_beams_key_function,
                              EOSTokenConfig, detect_eos_tokens_from_tokenizer,
                              get_beam_search_score)
from vllm.config import DecodingConfig, ModelConfig, VllmConfig
from vllm.core.scheduler import SchedulerOutputs
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.inputs.parse import is_explicit_encoder_decoder_prompt
from vllm.inputs.preprocess import InputPreprocessor
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Device, collect_from_async_generator, random_uuid

logger = init_logger(__name__)


class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        ...

    @property
    @abstractmethod
    def is_stopped(self) -> bool:
        ...

    @property
    @abstractmethod
    def errored(self) -> bool:
        ...

    @property
    @abstractmethod
    def dead_error(self) -> BaseException:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: PromptType,
        sampling_params: SamplingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...

    async def beam_search(
        self,
        prompt: PromptType,
        request_id: str,
        params: BeamSearchParams,
        lora_request: Optional[LoRARequest] = None,
    ) -> AsyncGenerator[RequestOutput, None]:

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        ignore_eos = params.ignore_eos
        temperature = params.temperature
        length_penalty = params.length_penalty
        include_stop_str_in_output = params.include_stop_str_in_output
        min_tokens = getattr(params, 'min_tokens', 0)
        additional_eos_token_ids = getattr(params, 'additional_eos_token_ids', []) or []

        preprocessor = await self.get_input_preprocessor()
        tokenizer_group = preprocessor.get_tokenizer_group()
        tokenizer = await tokenizer_group.get_lora_tokenizer_async()

        if is_explicit_encoder_decoder_prompt(prompt):
            raise NotImplementedError
        else:
            processed_inputs = preprocessor._prompt_to_llm_inputs(prompt)

        if processed_inputs["type"] == "embeds":
            raise NotImplementedError

        # This is a workaround to fix multimodal beam search; this is a
        # bandaid fix for 2 small problems:
        # 1. Multi_modal_data on the processed_inputs currently resolves to
        #    `None`.
        # 2. preprocessing above expands the multimodal placeholders. However,
        #    this happens again in generation, so the double expansion causes
        #    a mismatch.
        # TODO - would be ideal to handle this more gracefully.
        prompt_token_ids = prompt.get("prompt_token_ids")
        multi_modal_data = prompt.get("multi_modal_data")

        prompt_text = processed_inputs.get("prompt")
        mm_processor_kwargs = processed_inputs.get("mm_processor_kwargs")

        tokenized_length = len(prompt_token_ids)

        # Create comprehensive EOS configuration
        eos_config = detect_eos_tokens_from_tokenizer(tokenizer)
        eos_config.ignore_eos = ignore_eos
        eos_config.min_tokens = min_tokens
        eos_config.additional_eos_token_ids.update(additional_eos_token_ids)

        def sort_beams_key(beam: BeamSearchSequence) -> float:
            return get_beam_search_score(
                beam.tokens, beam.cum_logprob,
                eos_config.primary_eos_token_id or 0,
                length_penalty
            )

        beam_search_params = SamplingParams(
            logprobs=2 * beam_width,
            max_tokens=1,
            temperature=temperature,
        )
        all_beams = [
            BeamSearchSequence(tokens=prompt_token_ids,
                               cum_logprob=0,
                               logprobs=[],
                               multi_modal_data=multi_modal_data,
                               mm_processor_kwargs=mm_processor_kwargs,
                               lora_request=lora_request)
        ]
        completed = []
        current_step = 0

        for step in range(max_tokens):
            current_step = step
            
            if len(all_beams) == 0:
                break

            prompts_batch, lora_req_batch = zip(*[(
                TokensPrompt(prompt_token_ids=beam.tokens,
                             multi_modal_data=beam.multi_modal_data,
                             mm_processor_kwargs=beam.mm_processor_kwargs),
                beam.lora_request,
            ) for beam in all_beams])

            tasks = []

            step_request_id = f"beam_search-{random_uuid()}"
            for i, (individual_prompt,
                    lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
                request_id_item = f"{step_request_id}-{i}"
                task = asyncio.create_task(
                    collect_from_async_generator(
                        self.generate(individual_prompt,
                                      beam_search_params,
                                      request_id_item,
                                      lora_request=lora_req)))
                tasks.append(task)

            output = await asyncio.gather(*tasks)

            output = [x[0] for x in output]

            new_beams = []
            for i, current_beam in enumerate(all_beams):
                result = output[i]

                if result.outputs[0].logprobs is not None:
                    logprobs = result.outputs[0].logprobs[0]
                    for token_id, logprob_obj in logprobs.items():
                        new_beam = BeamSearchSequence(
                            tokens=current_beam.tokens + [token_id],
                            logprobs=current_beam.logprobs + [logprobs],
                            lora_request=current_beam.lora_request,
                            cum_logprob=current_beam.cum_logprob + logprob_obj.logprob,
                            multi_modal_data=current_beam.multi_modal_data,
                            mm_processor_kwargs=current_beam.mm_processor_kwargs
                        )
                        
                        # Enhanced EOS handling - process EOS for each beam
                        if eos_config.is_eos_token(token_id) and not ignore_eos:
                            # Check minimum token requirement (only count generated tokens, not prompt)
                            generated_tokens = len(new_beam.tokens) - tokenized_length
                            if generated_tokens >= min_tokens:
                                # Mark beam as finished and add to completed
                                new_beam.finish_reason = "stop"
                                new_beam.stop_reason = token_id
                                new_beam.is_finished = True
                                new_beam.finished_step = current_step
                                completed.append(new_beam)
                            else:
                                # Continue generation even with EOS if min_tokens not met
                                new_beams.append(new_beam)
                        else:
                            # Non-EOS token, continue beam
                            new_beams.append(new_beam)

            # Sort and keep top beams
            sorted_beams = sorted(new_beams, key=sort_beams_key, reverse=True)
            all_beams = sorted_beams[:beam_width]

        # Add any remaining active beams to completed
        for beam in all_beams:
            if not beam.is_finished:
                beam.finish_reason = "length"
                beam.is_finished = True
                beam.finished_step = current_step
            completed.append(beam)

        # Get best completed beams
        sorted_completed = sorted(completed, key=sort_beams_key, reverse=True)
        best_beams = sorted_completed[:beam_width]

        outputs = []
        for i, beam in enumerate(best_beams):
            # Determine which tokens to include in the output text and token_ids consistently
            if (beam.tokens and
                eos_config.is_eos_token(beam.tokens[-1]) and
                not ignore_eos and
                not include_stop_str_in_output):
                # Skip the eos token in both text and token_ids for consistency
                output_tokens = beam.tokens[tokenized_length:-1]
            else:
                output_tokens = beam.tokens[tokenized_length:]
            
            beam.text = tokenizer.decode(output_tokens)
            
            outputs.append(
                CompletionOutput(text=beam.text,
                                 cumulative_logprob=beam.cum_logprob,
                                 token_ids=output_tokens,
                                 index=i,
                                 logprobs=beam.logprobs,
                                 finish_reason=beam.finish_reason if
                                 beam.finish_reason is not None else "length",
                                 stop_reason=beam.stop_reason)
            )

        beam_search_output = RequestOutput(
            request_id=request_id,
            prompt=prompt_text,
            outputs=outputs,
            finished=True,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=None)

        yield beam_search_output

    @abstractmethod
    def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
        tokenization_kwargs: Optional[dict[str, Any]] = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        """Generate outputs for a request from a pooling model."""
        ...

    @abstractmethod
    async def abort(self, request_id: Union[str, Iterable[str]]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    @abstractmethod
    async def get_vllm_config(self) -> VllmConfig:
        """Get the vllm configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_model_config(self) -> ModelConfig:
        """Get the model configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_decoding_config(self) -> DecodingConfig:
        """Get the decoding configuration of the vLLM engine."""
        ...

    @abstractmethod
    async def get_input_preprocessor(self) -> InputPreprocessor:
        """Get the input processor of the vLLM engine."""
        ...

    @abstractmethod
    async def get_tokenizer(
        self,
        lora_request: Optional[LoRARequest] = None,
    ) -> AnyTokenizer:
        """Get the appropriate tokenizer for the request"""
        ...

    @abstractmethod
    async def is_tracing_enabled(self) -> bool:
        ...

    @abstractmethod
    async def do_log_stats(
        self,
        scheduler_outputs: Optional[SchedulerOutputs] = None,
        model_output: Optional[list[SamplerOutput]] = None,
    ) -> None:
        ...

    @abstractmethod
    async def check_health(self) -> None:
        """Raise if unhealthy"""
        ...

    @abstractmethod
    async def start_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def stop_profile(self) -> None:
        """Start profiling the engine"""
        ...

    @abstractmethod
    async def reset_mm_cache(self) -> None:
        """Reset the multi-modal cache"""
        ...

    @abstractmethod
    async def reset_prefix_cache(self,
                                 device: Optional[Device] = None) -> None:
        """Reset the prefix cache"""
        ...

    @abstractmethod
    async def sleep(self, level: int = 1) -> None:
        """Sleep the engine"""
        ...

    @abstractmethod
    async def wake_up(self, tags: Optional[list[str]] = None) -> None:
        """Wake up the engine"""
        ...

    @abstractmethod
    async def is_sleeping(self) -> bool:
        """Check whether the engine is sleeping"""
        ...

    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        ...

    async def scale_elastic_ep(self,
                               new_data_parallel_size: int,
                               drain_timeout: int = 300) -> None:
        """Scale the engine"""
        raise NotImplementedError

    async def collective_rpc(self,
                             method: str,
                             timeout: Optional[float] = None,
                             args: tuple = (),
                             kwargs: Optional[dict] = None):
        """Perform a collective RPC call to the given path."""
        raise NotImplementedError
