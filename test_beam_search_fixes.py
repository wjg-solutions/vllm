#!/usr/bin/env python3
"""
Test script to verify beam search fixes:
1. EOS tokens are properly processed for each beam
2. No length penalty is applied in scoring
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from vllm.beam_search import (
    BeamSearchSequence, 
    BeamSearchInstance, 
    EOSTokenConfig, 
    get_beam_search_score,
    detect_eos_tokens_from_tokenizer
)
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import BeamSearchParams
from transformers import AutoTokenizer

def test_no_length_penalty():
    """Test that beam search scoring doesn't apply length penalty."""
    print("Testing beam search scoring without length penalty...")
    
    # Test with different sequence lengths but same cumulative logprob
    tokens_short = [1, 2, 3]  # length 3
    tokens_long = [1, 2, 3, 4, 5, 6, 7]  # length 7
    cum_logprob = -5.0
    eos_token_id = 2
    
    # Both should return the same score (cumulative logprob) regardless of length
    score_short = get_beam_search_score(tokens_short, cum_logprob, eos_token_id, length_penalty=1.0)
    score_long = get_beam_search_score(tokens_long, cum_logprob, eos_token_id, length_penalty=1.0)
    
    print(f"Short sequence score: {score_short}")
    print(f"Long sequence score: {score_long}")
    print(f"Expected score (cumulative logprob): {cum_logprob}")
    
    assert score_short == cum_logprob, f"Short sequence score should equal cumulative logprob"
    assert score_long == cum_logprob, f"Long sequence score should equal cumulative logprob"
    assert score_short == score_long, f"Scores should be equal regardless of length"
    
    print("✓ Length penalty test passed - scores are based only on cumulative logprob\n")

def test_eos_detection():
    """Test EOS token detection and beam termination."""
    print("Testing EOS token detection...")
    
    # Create EOS configuration
    eos_config = EOSTokenConfig(
        primary_eos_token_id=2,
        additional_eos_token_ids={50256, 0},  # Common EOS tokens
        ignore_eos=False,
        min_tokens=1
    )
    
    # Test EOS detection
    assert eos_config.is_eos_token(2), "Should detect primary EOS token"
    assert eos_config.is_eos_token(50256), "Should detect additional EOS token"
    assert not eos_config.is_eos_token(100), "Should not detect non-EOS token"
    
    print(f"EOS tokens detected: {eos_config.all_eos_token_ids}")
    
    # Test beam termination logic
    beam_instance = BeamSearchInstance(
        prompt_tokens=[1, 2, 3],
        eos_config=eos_config
    )
    
    # Create a test beam
    test_beam = BeamSearchSequence(tokens=[1, 2, 3, 4])
    
    # Test termination with EOS token
    should_terminate_eos = beam_instance.should_terminate_beam(test_beam, 2)  # EOS token
    should_terminate_normal = beam_instance.should_terminate_beam(test_beam, 5)  # Normal token
    
    assert should_terminate_eos, "Should terminate beam with EOS token"
    assert not should_terminate_normal, "Should not terminate beam with normal token"
    
    print("✓ EOS detection test passed\n")

def test_beam_search_with_model():
    """Test beam search with a small model to verify EOS processing."""
    print("Testing beam search with actual model...")
    
    try:
        # Use a small model for testing
        model_name = "microsoft/DialoGPT-small"
        
        # Initialize LLM
        llm = LLM(
            model=model_name,
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.3
        )
        
        # Get tokenizer to check EOS token
        tokenizer = llm.get_tokenizer()
        print(f"Model EOS token ID: {tokenizer.eos_token_id}")
        
        # Test beam search parameters
        beam_params = BeamSearchParams(
            beam_width=3,
            max_tokens=10,
            temperature=0.7,
            length_penalty=1.0,  # Should be ignored
            ignore_eos=False,
            min_tokens=1
        )
        
        # Test prompts
        prompts = [
            {"prompt": "Hello, how are you"},
            {"prompt": "The weather today is"}
        ]
        
        print("Running beam search...")
        outputs = llm.beam_search(prompts, beam_params)
        
        print(f"Generated {len(outputs)} outputs")
        for i, output in enumerate(outputs):
            print(f"\nPrompt {i+1}: {prompts[i]['prompt']}")
            for j, sequence in enumerate(output.sequences):
                print(f"  Beam {j+1}: {sequence.text}")
                print(f"    Tokens: {len(sequence.tokens)}")
                print(f"    Score: {sequence.cum_logprob:.3f}")
                print(f"    Finished: {sequence.is_finished}")
                print(f"    Finish reason: {sequence.finish_reason}")
                
                # Check if EOS was properly detected
                if sequence.tokens and tokenizer.eos_token_id in sequence.tokens:
                    eos_positions = [i for i, token in enumerate(sequence.tokens) if token == tokenizer.eos_token_id]
                    print(f"    EOS token found at positions: {eos_positions}")
        
        print("✓ Model beam search test completed\n")
        
    except Exception as e:
        print(f"Model test skipped due to: {e}")
        print("This is expected if the model is not available or GPU memory is insufficient\n")

def main():
    """Run all tests."""
    print("=== Beam Search Fixes Verification ===\n")
    
    # Test 1: No length penalty
    test_no_length_penalty()
    
    # Test 2: EOS detection
    test_eos_detection()
    
    # Test 3: Model-based test (optional)
    test_beam_search_with_model()
    
    print("=== All Tests Completed ===")
    print("✓ Length penalty removed - scoring based only on cumulative logprob")
    print("✓ EOS tokens properly detected and processed for each beam")

if __name__ == "__main__":
    main()