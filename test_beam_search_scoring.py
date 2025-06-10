#!/usr/bin/env python3
"""
Test script to verify beam search scoring fixes.
This script tests the key issues that were identified and fixed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm.beam_search import (
    BeamSearchSequence, 
    get_beam_search_score, 
    EOSTokenConfig,
    BeamSearchInstance
)
from vllm.sequence import Logprob

def test_beam_search_score_consistency():
    """Test that beam search scoring is consistent with EOS token handling."""
    print("Testing beam search score consistency...")
    
    # Test case 1: Two sequences with same content but one has EOS token
    tokens_without_eos = [1, 2, 3, 4, 5]
    tokens_with_eos = [1, 2, 3, 4, 5, 2]  # 2 is EOS token
    cum_logprob = -10.5
    eos_token_id = 2
    length_penalty = 1.0
    
    score_without_eos = get_beam_search_score(tokens_without_eos, cum_logprob, eos_token_id, length_penalty)
    score_with_eos = get_beam_search_score(tokens_with_eos, cum_logprob, eos_token_id, length_penalty)
    
    print(f"Score without EOS: {score_without_eos}")
    print(f"Score with EOS: {score_with_eos}")
    
    # They should be equal since EOS token is excluded from length calculation
    assert abs(score_without_eos - score_with_eos) < 1e-6, "Scores should be equal when EOS is excluded"
    print("✅ EOS token exclusion test passed")

def test_beam_sequence_creation():
    """Test BeamSearchSequence creation and properties."""
    print("\nTesting BeamSearchSequence creation...")
    
    tokens = [1, 2, 3, 4]
    logprobs = [
        {1: Logprob(logprob=-1.0, rank=1, decoded_token="a")},
        {2: Logprob(logprob=-1.5, rank=1, decoded_token="b")},
        {3: Logprob(logprob=-2.0, rank=1, decoded_token="c")},
        {4: Logprob(logprob=-1.2, rank=1, decoded_token="d")}
    ]
    cum_logprob = -5.7
    
    beam = BeamSearchSequence(
        tokens=tokens,
        logprobs=logprobs,
        cum_logprob=cum_logprob
    )
    
    assert beam.tokens == tokens, "Tokens should match"
    assert beam.cum_logprob == cum_logprob, "Cumulative logprob should match"
    assert not beam.is_finished, "Beam should not be finished by default"
    print("✅ BeamSearchSequence creation test passed")

def test_eos_config():
    """Test EOS token configuration."""
    print("\nTesting EOS token configuration...")
    
    config = EOSTokenConfig(
        primary_eos_token_id=2,
        additional_eos_token_ids={3, 4},
        ignore_eos=False,
        min_tokens=2
    )
    
    assert config.is_eos_token(2), "Primary EOS token should be detected"
    assert config.is_eos_token(3), "Additional EOS token should be detected"
    assert config.is_eos_token(4), "Additional EOS token should be detected"
    assert not config.is_eos_token(5), "Non-EOS token should not be detected"
    
    # Test should_stop_at_eos
    tokens_short = [1, 2]  # Less than min_tokens
    tokens_long = [1, 5, 6, 2]  # More than min_tokens
    
    assert not config.should_stop_at_eos(tokens_short, 1), "Should not stop with too few tokens"
    assert config.should_stop_at_eos(tokens_long, 3), "Should stop with enough tokens"
    print("✅ EOS configuration test passed")

def test_beam_search_instance():
    """Test BeamSearchInstance functionality."""
    print("\nTesting BeamSearchInstance...")
    
    prompt_tokens = [1, 2, 3]
    eos_config = EOSTokenConfig(primary_eos_token_id=4, min_tokens=1)
    
    instance = BeamSearchInstance(
        prompt_tokens=prompt_tokens,
        eos_config=eos_config
    )
    
    assert len(instance.beams) == 1, "Should start with one beam"
    assert instance.beams[0].tokens == prompt_tokens, "Initial beam should have prompt tokens"
    assert len(instance.completed) == 0, "Should start with no completed beams"
    
    # Test beam termination
    beam = instance.beams[0]
    should_terminate = instance.should_terminate_beam(beam, 4)  # EOS token
    assert should_terminate, "Should terminate beam with EOS token"
    
    should_not_terminate = instance.should_terminate_beam(beam, 5)  # Non-EOS token
    assert not should_not_terminate, "Should not terminate beam with non-EOS token"
    
    print("✅ BeamSearchInstance test passed")

def test_length_penalty_effects():
    """Test that length penalty affects scoring correctly."""
    print("\nTesting length penalty effects...")
    
    tokens = [1, 2, 3, 4, 5]
    cum_logprob = -10.0
    eos_token_id = 99  # Not in tokens
    
    # Test different length penalties
    score_no_penalty = get_beam_search_score(tokens, cum_logprob, eos_token_id, 1.0)
    score_favor_long = get_beam_search_score(tokens, cum_logprob, eos_token_id, 1.5)
    score_favor_short = get_beam_search_score(tokens, cum_logprob, eos_token_id, 0.5)
    
    print(f"Score with no penalty (1.0): {score_no_penalty}")
    print(f"Score favoring long (1.5): {score_favor_long}")
    print(f"Score favoring short (0.5): {score_favor_short}")
    
    # Higher length penalty should give higher scores (favoring longer sequences)
    assert score_favor_long > score_no_penalty, "Higher length penalty should increase score"
    assert score_no_penalty > score_favor_short, "Lower length penalty should decrease score"
    print("✅ Length penalty test passed")

def main():
    """Run all tests."""
    print("Running beam search scoring tests...\n")
    
    try:
        test_beam_search_score_consistency()
        test_beam_sequence_creation()
        test_eos_config()
        test_beam_search_instance()
        test_length_penalty_effects()
        
        print("\n🎉 All tests passed! Beam search scoring fixes are working correctly.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())