# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, List

from vllm.beam_search import (
    BeamSearchInstance, BeamSearchSequence, EOSTokenConfig,
    detect_eos_tokens_from_tokenizer, detect_eos_tokens_from_generation_config,
    get_beam_search_score
)
from vllm.sampling_params import BeamSearchParams
from vllm.sequence import Logprob


class TestEOSTokenConfig:
    """Test EOS token configuration functionality."""
    
    def test_basic_eos_config(self):
        """Test basic EOS configuration."""
        config = EOSTokenConfig(primary_eos_token_id=2)
        assert config.primary_eos_token_id == 2
        assert config.is_eos_token(2)
        assert not config.is_eos_token(1)
    
    def test_additional_eos_tokens(self):
        """Test additional EOS tokens."""
        config = EOSTokenConfig(
            primary_eos_token_id=2,
            additional_eos_token_ids={3, 4}
        )
        assert config.is_eos_token(2)
        assert config.is_eos_token(3)
        assert config.is_eos_token(4)
        assert not config.is_eos_token(1)
    
    def test_all_eos_token_ids(self):
        """Test getting all EOS token IDs."""
        config = EOSTokenConfig(
            primary_eos_token_id=2,
            additional_eos_token_ids={3, 4}
        )
        all_eos = config.all_eos_token_ids
        assert all_eos == {2, 3, 4}
    
    def test_should_stop_at_eos_ignore_eos(self):
        """Test EOS stopping when ignore_eos is True."""
        config = EOSTokenConfig(
            primary_eos_token_id=2,
            ignore_eos=True
        )
        assert not config.should_stop_at_eos([1, 2], 1)
    
    def test_should_stop_at_eos_min_tokens(self):
        """Test EOS stopping with minimum token requirement."""
        config = EOSTokenConfig(
            primary_eos_token_id=2,
            min_tokens=5
        )
        # Should not stop if below min_tokens
        assert not config.should_stop_at_eos([1, 2], 1)
        # Should stop if above min_tokens
        assert config.should_stop_at_eos([1, 1, 1, 1, 1, 2], 1)


class TestBeamSearchInstance:
    """Test enhanced beam search instance functionality."""
    
    def create_mock_beam(self, tokens: List[int], cum_logprob: float = 0.0) -> BeamSearchSequence:
        """Create a mock beam sequence."""
        return BeamSearchSequence(
            tokens=tokens,
            logprobs=[],
            cum_logprob=cum_logprob
        )
    
    def test_should_terminate_beam(self):
        """Test beam termination logic."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        beam = self.create_mock_beam([1, 3])
        
        # Should terminate on EOS token
        assert instance.should_terminate_beam(beam, 2)
        # Should not terminate on non-EOS token
        assert not instance.should_terminate_beam(beam, 3)
    
    def test_should_terminate_beam_min_tokens(self):
        """Test beam termination with minimum tokens."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2, min_tokens=5)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        short_beam = self.create_mock_beam([1, 3])
        long_beam = self.create_mock_beam([1, 3, 4, 5, 6])
        
        # Should not terminate short beam even with EOS
        assert not instance.should_terminate_beam(short_beam, 2)
        # Should terminate long beam with EOS
        assert instance.should_terminate_beam(long_beam, 2)
    
    def test_finalize_beam(self):
        """Test beam finalization."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        beam = self.create_mock_beam([1, 3, 2])
        instance.finalize_beam(beam, "stop")
        
        assert beam.is_finished
        assert beam.finish_reason == "stop"
        assert beam.finished_step == 0
        assert beam.stop_reason == 2  # EOS token
    
    def test_add_completed_beam(self):
        """Test adding completed beams."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        beam = self.create_mock_beam([1, 3, 2])
        instance.add_completed_beam(beam)
        
        assert len(instance.completed) == 1
        assert beam.is_finished
        assert beam in instance.completed
    
    def test_cleanup_finished_beams(self):
        """Test cleanup of finished beams."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        # Add some beams
        finished_beam = self.create_mock_beam([1, 2])
        finished_beam.is_finished = True
        active_beam = self.create_mock_beam([1, 3])
        
        instance.beams = [finished_beam, active_beam]
        instance.cleanup_finished_beams()
        
        assert len(instance.beams) == 1
        assert active_beam in instance.beams
        assert finished_beam not in instance.beams
    
    def test_get_best_completed_beams(self):
        """Test getting best completed beams."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        # Add completed beams with different scores
        beam1 = self.create_mock_beam([1, 2], cum_logprob=-1.0)
        beam2 = self.create_mock_beam([1, 3, 2], cum_logprob=-0.5)
        beam3 = self.create_mock_beam([1, 4, 2], cum_logprob=-2.0)
        
        instance.completed = [beam1, beam2, beam3]
        
        best_beams = instance.get_best_completed_beams(beam_width=2)
        assert len(best_beams) == 2
        # beam2 should be first (highest score)
        assert best_beams[0] == beam2
        assert best_beams[1] == beam1
    
    def test_should_early_stop(self):
        """Test early stopping logic."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        # Not enough completed beams
        beam1 = self.create_mock_beam([1, 2], cum_logprob=-1.0)
        instance.completed = [beam1]
        assert not instance.should_early_stop(beam_width=2)
        
        # Enough completed beams, but active beam might be better
        beam2 = self.create_mock_beam([1, 3, 2], cum_logprob=-0.5)
        instance.completed = [beam1, beam2]
        active_beam = self.create_mock_beam([1, 3], cum_logprob=-0.1)  # Better score
        instance.beams = [active_beam]
        assert not instance.should_early_stop(beam_width=2)
        
        # Should stop when no active beam can beat completed beams
        active_beam.cum_logprob = -3.0  # Worse score
        assert instance.should_early_stop(beam_width=2)


class TestEOSTokenDetection:
    """Test EOS token detection from various sources."""
    
    def test_detect_eos_from_tokenizer_basic(self):
        """Test basic EOS detection from tokenizer."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.special_tokens_map = {}
        
        config = detect_eos_tokens_from_tokenizer(mock_tokenizer)
        assert config.primary_eos_token_id == 2
    
    def test_detect_eos_from_tokenizer_special_tokens(self):
        """Test EOS detection with special tokens."""
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.special_tokens_map = {
            'eos_token': '</s>',
            'additional_special_tokens': ['<|endoftext|>']
        }
        mock_tokenizer.convert_tokens_to_ids = Mock(side_effect=lambda x: {'</s>': 2, '<|endoftext|>': 50256}.get(x))
        mock_tokenizer.get_vocab = Mock(return_value={'</s>': 2, '<|endoftext|>': 50256})
        
        config = detect_eos_tokens_from_tokenizer(mock_tokenizer)
        assert config.primary_eos_token_id == 2
        assert 50256 in config.additional_eos_token_ids
    
    def test_detect_eos_from_generation_config(self):
        """Test EOS detection from generation config."""
        mock_config = Mock()
        mock_config.eos_token_id = [2, 3, 4]
        
        eos_tokens = detect_eos_tokens_from_generation_config(mock_config)
        assert eos_tokens == {2, 3, 4}
        
        # Test single EOS token
        mock_config.eos_token_id = 2
        eos_tokens = detect_eos_tokens_from_generation_config(mock_config)
        assert eos_tokens == {2}


class TestBeamSearchParams:
    """Test enhanced beam search parameters."""
    
    def test_beam_search_params_defaults(self):
        """Test default beam search parameters."""
        params = BeamSearchParams(beam_width=4, max_tokens=50)
        assert params.beam_width == 4
        assert params.max_tokens == 50
        assert params.ignore_eos is False
        assert params.min_tokens == 0
        assert params.early_stopping is True
        assert params.additional_eos_token_ids is None
        assert params.eos_token_penalty == 0.0
    
    def test_beam_search_params_custom(self):
        """Test custom beam search parameters."""
        params = BeamSearchParams(
            beam_width=8,
            max_tokens=100,
            ignore_eos=True,
            min_tokens=10,
            early_stopping=False,
            additional_eos_token_ids=[50256, 50257],
            eos_token_penalty=0.5
        )
        assert params.beam_width == 8
        assert params.max_tokens == 100
        assert params.ignore_eos is True
        assert params.min_tokens == 10
        assert params.early_stopping is False
        assert params.additional_eos_token_ids == [50256, 50257]
        assert params.eos_token_penalty == 0.5


class TestBeamSearchScoring:
    """Test beam search scoring with EOS handling."""
    
    def test_get_beam_search_score_with_eos(self):
        """Test beam search scoring with EOS token."""
        tokens_with_eos = [1, 2, 3, 2]  # EOS token is 2
        tokens_without_eos = [1, 2, 3, 4]
        cum_logprob = -2.0
        eos_token_id = 2
        length_penalty = 1.0
        
        score_with_eos = get_beam_search_score(tokens_with_eos, cum_logprob, eos_token_id, length_penalty)
        score_without_eos = get_beam_search_score(tokens_without_eos, cum_logprob, eos_token_id, length_penalty)
        
        # Score with EOS should exclude the EOS token from length calculation
        expected_score_with_eos = cum_logprob / 3  # Length 4 - 1 (EOS)
        expected_score_without_eos = cum_logprob / 4  # Full length
        
        assert abs(score_with_eos - expected_score_with_eos) < 1e-6
        assert abs(score_without_eos - expected_score_without_eos) < 1e-6
    
    def test_get_beam_search_score_length_penalty(self):
        """Test beam search scoring with length penalty."""
        tokens = [1, 2, 3, 4]
        cum_logprob = -2.0
        eos_token_id = 999  # Not in tokens
        length_penalty = 2.0
        
        score = get_beam_search_score(tokens, cum_logprob, eos_token_id, length_penalty)
        expected_score = cum_logprob / (4 ** length_penalty)
        
        assert abs(score - expected_score) < 1e-6


@pytest.mark.integration
class TestBeamSearchIntegration:
    """Integration tests for beam search with EOS handling."""
    
    def test_beam_search_early_termination(self):
        """Test that beam search terminates early when all beams hit EOS."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        # Simulate all beams hitting EOS
        beam1 = BeamSearchSequence(tokens=[1, 2], logprobs=[], cum_logprob=-1.0)
        beam2 = BeamSearchSequence(tokens=[1, 3, 2], logprobs=[], cum_logprob=-1.5)
        
        instance.finalize_beam(beam1)
        instance.finalize_beam(beam2)
        instance.completed = [beam1, beam2]
        instance.beams = []  # No active beams
        
        # Should early stop since no active beams remain
        assert instance.should_early_stop(beam_width=2)
    
    def test_beam_search_min_tokens_enforcement(self):
        """Test that minimum tokens are enforced before EOS termination."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2, min_tokens=5)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        # Short sequence with EOS should not terminate
        short_beam = BeamSearchSequence(tokens=[1, 3], logprobs=[], cum_logprob=-1.0)
        assert not instance.should_terminate_beam(short_beam, 2)
        
        # Long sequence with EOS should terminate
        long_beam = BeamSearchSequence(tokens=[1, 3, 4, 5, 6], logprobs=[], cum_logprob=-1.0)
        assert instance.should_terminate_beam(long_beam, 2)
    
    def test_beam_search_memory_cleanup(self):
        """Test that finished beams are properly cleaned up."""
        eos_config = EOSTokenConfig(primary_eos_token_id=2)
        instance = BeamSearchInstance([1], eos_config=eos_config)
        
        # Create mix of finished and active beams
        finished_beam = BeamSearchSequence(tokens=[1, 2], logprobs=[], cum_logprob=-1.0, is_finished=True)
        active_beam1 = BeamSearchSequence(tokens=[1, 3], logprobs=[], cum_logprob=-1.5)
        active_beam2 = BeamSearchSequence(tokens=[1, 4], logprobs=[], cum_logprob=-2.0)
        
        instance.beams = [finished_beam, active_beam1, active_beam2]
        initial_count = len(instance.beams)
        
        instance.cleanup_finished_beams()
        
        # Should have removed finished beam
        assert len(instance.beams) == initial_count - 1
        assert finished_beam not in instance.beams
        assert active_beam1 in instance.beams
        assert active_beam2 in instance.beams


if __name__ == "__main__":
    pytest.main([__file__])