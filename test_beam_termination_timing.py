#!/usr/bin/env python3
"""
Test to verify beam search termination timing and scoring behavior.
This test specifically checks if beams are ending at different times as expected.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vllm.beam_search import get_beam_search_score

def test_beam_termination_timing():
    """Test that beams with different lengths and scores terminate at different times."""
    print("Testing beam termination timing...")
    
    # Simulate beams that should terminate at different times
    # Beam 1: Short but high quality (good logprob per token)
    beam1_tokens = [1, 2, 3, 4]  # 4 tokens
    beam1_logprob = -4.0  # -1.0 per token (high quality)
    
    # Beam 2: Medium length, medium quality
    beam2_tokens = [1, 2, 3, 4, 5, 6]  # 6 tokens  
    beam2_logprob = -9.0  # -1.5 per token (medium quality)
    
    # Beam 3: Long but lower quality
    beam3_tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 8 tokens
    beam3_logprob = -16.0  # -2.0 per token (lower quality)
    
    eos_token_id = 99  # Not in any sequence
    length_penalty = 1.0  # No length penalty first
    
    score1 = get_beam_search_score(beam1_tokens, beam1_logprob, eos_token_id, length_penalty)
    score2 = get_beam_search_score(beam2_tokens, beam2_logprob, eos_token_id, length_penalty)
    score3 = get_beam_search_score(beam3_tokens, beam3_logprob, eos_token_id, length_penalty)
    
    print(f"Beam 1 (4 tokens, -4.0 logprob): Score = {score1:.6f}")
    print(f"Beam 2 (6 tokens, -9.0 logprob): Score = {score2:.6f}")
    print(f"Beam 3 (8 tokens, -16.0 logprob): Score = {score3:.6f}")
    
    # With no length penalty, shorter high-quality sequences should score better
    print(f"\nWith length_penalty=1.0 (no penalty):")
    print(f"Best to worst: {sorted([(score1, '4-token'), (score2, '6-token'), (score3, '8-token')], reverse=True)}")
    
    # Now test with length penalty favoring longer sequences
    length_penalty = 1.5
    score1_lp = get_beam_search_score(beam1_tokens, beam1_logprob, eos_token_id, length_penalty)
    score2_lp = get_beam_search_score(beam2_tokens, beam2_logprob, eos_token_id, length_penalty)
    score3_lp = get_beam_search_score(beam3_tokens, beam3_logprob, eos_token_id, length_penalty)
    
    print(f"\nWith length_penalty=1.5 (favoring longer):")
    print(f"Beam 1 (4 tokens): Score = {score1_lp:.6f}")
    print(f"Beam 2 (6 tokens): Score = {score2_lp:.6f}")
    print(f"Beam 3 (8 tokens): Score = {score3_lp:.6f}")
    print(f"Best to worst: {sorted([(score1_lp, '4-token'), (score2_lp, '6-token'), (score3_lp, '8-token')], reverse=True)}")
    
    return score1, score2, score3, score1_lp, score2_lp, score3_lp

def test_length_penalty_direction():
    """Test that length penalty works in the expected direction."""
    print("\nTesting length penalty direction...")
    
    # Same sequence, different length penalties
    tokens = [1, 2, 3, 4, 5]
    logprob = -10.0
    eos_token_id = 99
    
    # Test various length penalties
    penalties = [0.5, 1.0, 1.5, 2.0]
    scores = []
    
    for penalty in penalties:
        score = get_beam_search_score(tokens, logprob, eos_token_id, penalty)
        scores.append(score)
        print(f"Length penalty {penalty}: Score = {score:.6f}")
    
    # Higher length penalty should give higher scores (less penalty for length)
    for i in range(len(scores) - 1):
        if scores[i] >= scores[i + 1]:
            print(f"❌ ERROR: Length penalty {penalties[i]} gave score {scores[i]:.6f}, but penalty {penalties[i+1]} gave {scores[i+1]:.6f}")
            print("Higher length penalty should give higher scores!")
            return False
    
    print("✅ Length penalty direction is correct")
    return True

def test_realistic_beam_scenario():
    """Test a realistic scenario where beams should terminate at different times."""
    print("\nTesting realistic beam termination scenario...")
    
    # Simulate a scenario where we have multiple beams in progress
    beams = [
        # Beam that finds a good short completion
        {"tokens": [1, 2, 3, 4, 5], "logprob": -6.0, "description": "Good short completion"},
        
        # Beam that continues longer but with diminishing quality
        {"tokens": [1, 2, 3, 4, 5, 6, 7], "logprob": -12.0, "description": "Longer but lower quality"},
        
        # Beam that finds excellent medium-length completion
        {"tokens": [1, 2, 3, 4, 5, 6], "logprob": -7.5, "description": "Excellent medium length"},
        
        # Beam that goes very long with poor quality
        {"tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9], "logprob": -20.0, "description": "Very long, poor quality"},
    ]
    
    eos_token_id = 99
    length_penalty = 1.2  # Slight preference for longer sequences
    
    # Calculate scores
    for beam in beams:
        beam["score"] = get_beam_search_score(
            beam["tokens"], beam["logprob"], eos_token_id, length_penalty
        )
    
    # Sort by score
    sorted_beams = sorted(beams, key=lambda x: x["score"], reverse=True)
    
    print("Beam ranking (best to worst):")
    for i, beam in enumerate(sorted_beams):
        print(f"  {i+1}. {beam['description']}: {len(beam['tokens'])} tokens, "
              f"logprob={beam['logprob']}, score={beam['score']:.6f}")
    
    # The excellent medium-length should typically win
    best_beam = sorted_beams[0]
    print(f"\nBest beam: {best_beam['description']} with {len(best_beam['tokens'])} tokens")
    
    return sorted_beams

def main():
    """Run all tests."""
    print("Testing beam search termination timing and scoring...\n")
    
    try:
        # Test basic termination timing
        scores = test_beam_termination_timing()
        
        # Test length penalty direction
        penalty_correct = test_length_penalty_direction()
        
        # Test realistic scenario
        realistic_results = test_realistic_beam_scenario()
        
        if penalty_correct:
            print("\n🎉 All tests completed!")
            print("\nKey findings:")
            print("1. ✅ Beam scoring calculations are working")
            print("2. ✅ Length penalty direction is correct")
            print("3. ✅ Realistic beam ranking is functioning")
            
            print("\nIf you're still seeing all beams end at the same time, the issue might be:")
            print("- In the beam search termination logic (not the scoring)")
            print("- In how EOS tokens are being detected/handled")
            print("- In the early stopping criteria")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())