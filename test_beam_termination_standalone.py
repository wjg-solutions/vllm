#!/usr/bin/env python3
"""
Standalone test to verify beam search termination timing and scoring behavior.
This test specifically checks if beams are ending at different times as expected.
"""

def get_beam_search_score(tokens, cumulative_logprob, eos_token_id, length_penalty=1.0):
    """
    Beam search scoring function (copied from vllm/beam_search.py to avoid dependencies).
    """
    seq_len = len(tokens)
    adjusted_logprob = cumulative_logprob
    
    # If sequence ends with EOS token, exclude it from length calculation
    if tokens and tokens[-1] == eos_token_id:
        seq_len -= 1
    
    # GNMT-style length penalty (Wu et al., 2016)
    alpha = length_penalty
    k = 5.0
    lp = ((k + seq_len) ** alpha) / ((k + 1) ** alpha)
    return adjusted_logprob / lp

def test_length_penalty_direction():
    """Test that length penalty works in the expected direction."""
    print("Testing length penalty direction...")
    
    # Same sequence, different length penalties
    tokens = [1, 2, 3, 4, 5]
    logprob = -10.0
    eos_token_id = 99
    
    # Test various length penalties
    penalties = [0.5, 1.0, 1.5, 2.0]
    scores = []
    
    print("Length penalty effects:")
    for penalty in penalties:
        score = get_beam_search_score(tokens, logprob, eos_token_id, penalty)
        scores.append(score)
        print(f"  Length penalty {penalty}: Score = {score:.6f}")
    
    # Check if higher length penalty gives higher scores
    print(f"\nScore progression: {[f'{s:.3f}' for s in scores]}")
    
    # Higher length penalty should give higher scores (less penalty for length)
    issues = []
    for i in range(len(scores) - 1):
        if scores[i] >= scores[i + 1]:
            issues.append(f"Length penalty {penalties[i]} gave score {scores[i]:.6f}, but penalty {penalties[i+1]} gave {scores[i+1]:.6f}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print("Higher length penalty should give higher scores!")
        return False, scores
    
    print("✅ Length penalty direction is correct")
    return True, scores

def test_beam_ranking_with_different_lengths():
    """Test beam ranking with sequences of different lengths."""
    print("\nTesting beam ranking with different lengths...")
    
    # Create beams with different characteristics
    beams = [
        # Short but high quality (good logprob per token)
        {"tokens": [1, 2, 3, 4], "logprob": -4.0, "desc": "Short, high quality"},
        
        # Medium length, medium quality  
        {"tokens": [1, 2, 3, 4, 5, 6], "logprob": -9.0, "desc": "Medium length, medium quality"},
        
        # Long but lower quality
        {"tokens": [1, 2, 3, 4, 5, 6, 7, 8], "logprob": -16.0, "desc": "Long, lower quality"},
        
        # Very short, excellent quality
        {"tokens": [1, 2], "logprob": -1.5, "desc": "Very short, excellent"},
        
        # Medium-long, good quality
        {"tokens": [1, 2, 3, 4, 5, 6, 7], "logprob": -10.5, "desc": "Medium-long, good quality"},
    ]
    
    eos_token_id = 99
    
    # Test with different length penalties
    for length_penalty in [0.8, 1.0, 1.2]:
        print(f"\nWith length_penalty = {length_penalty}:")
        
        # Calculate scores
        for beam in beams:
            beam["score"] = get_beam_search_score(
                beam["tokens"], beam["logprob"], eos_token_id, length_penalty
            )
        
        # Sort by score (best first)
        sorted_beams = sorted(beams, key=lambda x: x["score"], reverse=True)
        
        print("  Ranking (best to worst):")
        for i, beam in enumerate(sorted_beams):
            avg_logprob = beam["logprob"] / len(beam["tokens"])
            print(f"    {i+1}. {beam['desc']}: {len(beam['tokens'])} tokens, "
                  f"total_logprob={beam['logprob']:.1f}, avg={avg_logprob:.2f}, score={beam['score']:.4f}")
    
    return True

def test_potential_scoring_issue():
    """Test for the specific issue: shorter sequences getting unfairly high scores."""
    print("\nTesting for potential scoring bias toward shorter sequences...")
    
    # Create a scenario where a longer sequence should win but might not
    beams = [
        # Short sequence that ends early (might be getting unfair advantage)
        {"tokens": [1, 2, 3], "logprob": -4.5, "desc": "Short early termination"},
        
        # Longer sequence with better overall quality
        {"tokens": [1, 2, 3, 4, 5, 6, 7], "logprob": -10.5, "desc": "Longer, better content"},
        
        # Medium sequence with good quality
        {"tokens": [1, 2, 3, 4, 5], "logprob": -7.5, "desc": "Medium, good quality"},
    ]
    
    eos_token_id = 99
    length_penalty = 1.0  # No length bias
    
    print("Analyzing potential bias (length_penalty = 1.0):")
    
    for beam in beams:
        beam["score"] = get_beam_search_score(
            beam["tokens"], beam["logprob"], eos_token_id, length_penalty
        )
        avg_logprob = beam["logprob"] / len(beam["tokens"])
        print(f"  {beam['desc']}: {len(beam['tokens'])} tokens, "
              f"avg_logprob={avg_logprob:.3f}, score={beam['score']:.4f}")
    
    # Sort by score
    sorted_beams = sorted(beams, key=lambda x: x["score"], reverse=True)
    winner = sorted_beams[0]
    
    print(f"\nWinner: {winner['desc']}")
    
    # Check if the winner makes sense
    if len(winner["tokens"]) <= 3 and len(sorted_beams) > 1:
        second_place = sorted_beams[1]
        winner_avg = winner["logprob"] / len(winner["tokens"])
        second_avg = second_place["logprob"] / len(second_place["tokens"])
        
        print(f"⚠️  WARNING: Very short sequence won!")
        print(f"   Winner avg logprob: {winner_avg:.3f}")
        print(f"   Second place avg logprob: {second_avg:.3f}")
        
        if second_avg > winner_avg:
            print(f"   🔍 Second place has better average quality but lost due to length penalty")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Testing beam search termination timing and scoring behavior...\n")
    
    try:
        # Test length penalty direction
        penalty_correct, penalty_scores = test_length_penalty_direction()
        
        # Test beam ranking with different lengths
        ranking_test = test_beam_ranking_with_different_lengths()
        
        # Test for potential scoring bias
        bias_test = test_potential_scoring_issue()
        
        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        
        if penalty_correct:
            print("✅ Length penalty direction: CORRECT")
        else:
            print("❌ Length penalty direction: INCORRECT")
            print("   This could cause beams to terminate at wrong times!")
        
        if ranking_test:
            print("✅ Beam ranking: Working as expected")
        else:
            print("❌ Beam ranking: Issues detected")
        
        if bias_test:
            print("✅ No obvious bias toward short sequences")
        else:
            print("❌ Potential bias toward short sequences detected")
            print("   This could explain why you're getting shorter responses!")
        
        print("\nIf you're still seeing all beams end at the same time:")
        print("1. Check the beam search termination logic (not just scoring)")
        print("2. Verify EOS token detection is working correctly")
        print("3. Look at the early stopping criteria")
        print("4. Check if length penalty is being applied correctly in the actual beam search")
        
        if not penalty_correct:
            print("\n🚨 CRITICAL: Length penalty direction is wrong!")
            print("   This means higher length penalties are making scores WORSE, not better.")
            print("   This could cause all beams to prefer shorter sequences.")
        
        return 0 if penalty_correct and ranking_test and bias_test else 1
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())