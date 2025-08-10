#!/usr/bin/env python3
"""
Simple test script to verify beam search fixes without requiring torch:
1. EOS tokens are properly processed for each beam
2. No length penalty is applied in scoring
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_beam_search_score_no_length_penalty():
    """Test that beam search scoring doesn't apply length penalty."""
    print("Testing beam search scoring without length penalty...")
    
    # Import the scoring function directly
    try:
        from vllm.beam_search import get_beam_search_score
    except ImportError as e:
        print(f"Could not import beam search module: {e}")
        return False
    
    # Test with different sequence lengths but same cumulative logprob
    tokens_short = [1, 2, 3]  # length 3
    tokens_long = [1, 2, 3, 4, 5, 6, 7]  # length 7
    cum_logprob = -5.0
    eos_token_id = 2
    
    # Both should return the same score (cumulative logprob) regardless of length
    score_short = get_beam_search_score(tokens_short, cum_logprob, eos_token_id, length_penalty=1.0)
    score_long = get_beam_search_score(tokens_long, cum_logprob, eos_token_id, length_penalty=1.0)
    
    print(f"Short sequence (length {len(tokens_short)}) score: {score_short}")
    print(f"Long sequence (length {len(tokens_long)}) score: {score_long}")
    print(f"Expected score (cumulative logprob): {cum_logprob}")
    
    # Verify scores match cumulative logprob
    if abs(score_short - cum_logprob) < 1e-6:
        print("✓ Short sequence score matches cumulative logprob")
    else:
        print(f"✗ Short sequence score mismatch: {score_short} != {cum_logprob}")
        return False
    
    if abs(score_long - cum_logprob) < 1e-6:
        print("✓ Long sequence score matches cumulative logprob")
    else:
        print(f"✗ Long sequence score mismatch: {score_long} != {cum_logprob}")
        return False
    
    if abs(score_short - score_long) < 1e-6:
        print("✓ Scores are equal regardless of sequence length")
    else:
        print(f"✗ Scores differ by length: {score_short} != {score_long}")
        return False
    
    print("✓ Length penalty test passed - scores are based only on cumulative logprob\n")
    return True

def test_eos_config():
    """Test EOS token configuration without requiring full imports."""
    print("Testing EOS token configuration...")
    
    try:
        from vllm.beam_search import EOSTokenConfig
    except ImportError as e:
        print(f"Could not import EOS config: {e}")
        return False
    
    # Create EOS configuration
    eos_config = EOSTokenConfig(
        primary_eos_token_id=2,
        additional_eos_token_ids={50256, 0},  # Common EOS tokens
        ignore_eos=False,
        min_tokens=1
    )
    
    # Test EOS detection
    tests = [
        (2, True, "primary EOS token"),
        (50256, True, "additional EOS token"),
        (0, True, "additional EOS token (0)"),
        (100, False, "non-EOS token"),
        (999, False, "another non-EOS token")
    ]
    
    all_passed = True
    for token_id, expected, description in tests:
        result = eos_config.is_eos_token(token_id)
        if result == expected:
            print(f"✓ {description} detection: {token_id} -> {result}")
        else:
            print(f"✗ {description} detection failed: {token_id} -> {result}, expected {expected}")
            all_passed = False
    
    print(f"All EOS tokens: {eos_config.all_eos_token_ids}")
    
    # Test should_stop_at_eos logic
    test_tokens = [1, 2, 3, 4, 5]  # 5 tokens
    should_stop = eos_config.should_stop_at_eos(test_tokens, current_step=3)
    
    if should_stop:
        print("✓ Should stop at EOS when conditions are met")
    else:
        print("✗ Should stop at EOS logic failed")
        all_passed = False
    
    # Test with ignore_eos=True
    eos_config.ignore_eos = True
    should_stop_ignored = eos_config.should_stop_at_eos(test_tokens, current_step=3)
    
    if not should_stop_ignored:
        print("✓ Should not stop when ignore_eos=True")
    else:
        print("✗ Should not stop when ignore_eos=True, but it did")
        all_passed = False
    
    if all_passed:
        print("✓ EOS configuration test passed\n")
    else:
        print("✗ EOS configuration test failed\n")
    
    return all_passed

def test_beam_instance_logic():
    """Test BeamSearchInstance logic without requiring full imports."""
    print("Testing BeamSearchInstance logic...")
    
    try:
        from vllm.beam_search import BeamSearchInstance, BeamSearchSequence, EOSTokenConfig
    except ImportError as e:
        print(f"Could not import beam search classes: {e}")
        return False
    
    # Create EOS configuration
    eos_config = EOSTokenConfig(
        primary_eos_token_id=2,
        ignore_eos=False,
        min_tokens=1
    )
    
    # Create beam instance
    beam_instance = BeamSearchInstance(
        prompt_tokens=[1, 2, 3],
        eos_config=eos_config
    )
    
    # Create test beam
    test_beam = BeamSearchSequence(tokens=[1, 2, 3, 4])
    
    # Test termination logic
    should_terminate_eos = beam_instance.should_terminate_beam(test_beam, 2)  # EOS token
    should_terminate_normal = beam_instance.should_terminate_beam(test_beam, 5)  # Normal token
    
    if should_terminate_eos:
        print("✓ Should terminate beam with EOS token")
    else:
        print("✗ Should terminate beam with EOS token, but didn't")
        return False
    
    if not should_terminate_normal:
        print("✓ Should not terminate beam with normal token")
    else:
        print("✗ Should not terminate beam with normal token, but did")
        return False
    
    # Test beam finalization
    new_beam = BeamSearchSequence(tokens=[1, 2, 3, 4, 2])  # Ends with EOS
    beam_instance.finalize_beam(new_beam, "stop")
    
    if new_beam.is_finished:
        print("✓ Beam properly finalized")
    else:
        print("✗ Beam finalization failed")
        return False
    
    if new_beam.finish_reason == "stop":
        print("✓ Finish reason set correctly")
    else:
        print(f"✗ Finish reason incorrect: {new_beam.finish_reason}")
        return False
    
    print("✓ BeamSearchInstance logic test passed\n")
    return True

def main():
    """Run all tests."""
    print("=== Beam Search Fixes Verification (Simple) ===\n")
    
    all_tests_passed = True
    
    # Test 1: No length penalty
    if not test_beam_search_score_no_length_penalty():
        all_tests_passed = False
    
    # Test 2: EOS configuration
    if not test_eos_config():
        all_tests_passed = False
    
    # Test 3: Beam instance logic
    if not test_beam_instance_logic():
        all_tests_passed = False
    
    print("=== Test Results ===")
    if all_tests_passed:
        print("✓ All tests passed!")
        print("✓ Length penalty removed - scoring based only on cumulative logprob")
        print("✓ EOS tokens properly detected and processed for each beam")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)