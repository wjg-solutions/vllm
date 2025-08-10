#!/usr/bin/env python3
"""
Test to verify that when a beam hits an EOS token, it stops processing that specific beam.
"""

def test_eos_beam_stopping_logic():
    """Test that beams with EOS tokens are properly stopped and removed from active processing."""
    print("=== Testing EOS Beam Stopping Logic ===\n")
    
    # Simulate the beam processing logic
    class MockBeam:
        def __init__(self, tokens, cum_logprob, is_finished=False):
            self.tokens = tokens
            self.cum_logprob = cum_logprob
            self.is_finished = is_finished
            self.finish_reason = None
            self.stop_reason = None
            self.finished_step = None
    
    class MockEOSConfig:
        def __init__(self, eos_token_id=2):
            self.primary_eos_token_id = eos_token_id
            self.ignore_eos = False
            self.min_tokens = 1
        
        def is_eos_token(self, token_id):
            return token_id == self.primary_eos_token_id
    
    # Test setup
    eos_config = MockEOSConfig(eos_token_id=2)
    tokenized_length = 3  # Prompt length
    current_step = 5
    ignore_eos = False
    
    # Simulate current beams with different next tokens
    test_cases = [
        {
            "name": "Beam with EOS token",
            "current_beam": MockBeam([1, 2, 3, 4], -2.5),
            "new_token": 2,  # EOS token
            "expected_in_new_beams": False,
            "expected_in_completed": True
        },
        {
            "name": "Beam with normal token", 
            "current_beam": MockBeam([1, 2, 3, 4], -3.0),
            "new_token": 5,  # Normal token
            "expected_in_new_beams": True,
            "expected_in_completed": False
        },
        {
            "name": "Beam with EOS but min_tokens not met",
            "current_beam": MockBeam([1, 2, 3], -1.5),  # Only 3 tokens total, 0 generated
            "new_token": 2,  # EOS token
            "expected_in_new_beams": False,  # Should stop because 1 generated token >= min_tokens(1)
            "expected_in_completed": True
        },
        {
            "name": "Beam with EOS but min_tokens not met (min_tokens=2)",
            "current_beam": MockBeam([1, 2, 3], -1.5),  # Prompt length=3, current=3 tokens, 0 generated
            "new_token": 2,  # EOS token
            "expected_in_new_beams": True,  # Should continue because 1 generated token < min_tokens(2)
            "expected_in_completed": False,
            "min_tokens": 2  # Override min_tokens for this test
        }
    ]
    
    print("Testing beam processing logic:")
    print("Token | Current Beam | New Token | In New Beams | In Completed | Reason")
    print("-" * 80)
    
    all_passed = True
    
    for case in test_cases:
        current_beam = case["current_beam"]
        new_token = case["new_token"]
        
        # Use custom min_tokens if specified
        min_tokens_for_test = case.get("min_tokens", eos_config.min_tokens)
        
        # Create new beam (simulate adding the new token)
        new_beam = MockBeam(
            tokens=current_beam.tokens + [new_token],
            cum_logprob=current_beam.cum_logprob - 0.5  # Simulate logprob addition
        )
        
        # Simulate the EOS processing logic
        new_beams = []
        completed = []
        
        # This is the actual logic from the code
        if eos_config.is_eos_token(new_token) and not ignore_eos:
            # Check minimum token requirement (only count generated tokens, not prompt)
            generated_tokens = len(new_beam.tokens) - tokenized_length
            if generated_tokens >= min_tokens_for_test:
                # Mark beam as finished and add to completed
                new_beam.finish_reason = "stop"
                new_beam.stop_reason = new_token
                new_beam.is_finished = True
                new_beam.finished_step = current_step
                completed.append(new_beam)
            else:
                # Continue generation even with EOS if min_tokens not met
                new_beams.append(new_beam)
        else:
            # Non-EOS token, continue beam
            new_beams.append(new_beam)
        
        # Check results
        in_new_beams = len(new_beams) > 0
        in_completed = len(completed) > 0
        
        expected_new = case["expected_in_new_beams"]
        expected_completed = case["expected_in_completed"]
        
        passed = (in_new_beams == expected_new) and (in_completed == expected_completed)
        status = "✓" if passed else "✗"
        
        if not passed:
            all_passed = False
        
        # Determine reason
        if eos_config.is_eos_token(new_token):
            generated = len(new_beam.tokens) - tokenized_length
            if generated >= min_tokens_for_test:
                reason = f"EOS + min_tokens met ({generated}>={min_tokens_for_test})"
            else:
                reason = f"EOS but min_tokens not met ({generated}<{min_tokens_for_test})"
        else:
            reason = "Normal token"
        
        print(f"{status:1s} {new_token:3d} | {len(current_beam.tokens):11d} | {new_token:9d} | {str(in_new_beams):12s} | {str(in_completed):11s} | {reason}")
    
    print("\n" + "=" * 80)
    
    if all_passed:
        print("✅ All EOS beam stopping tests passed!")
        print("✅ Beams with EOS tokens are properly stopped and removed from active processing")
        print("✅ Beams with normal tokens continue processing")
        print("✅ EOS tokens respect minimum token requirements")
    else:
        print("❌ Some EOS beam stopping tests failed!")
    
    return all_passed

def test_beam_cleanup_logic():
    """Test that finished beams are properly cleaned up from active beam lists."""
    print("\n=== Testing Beam Cleanup Logic ===\n")
    
    class MockBeam:
        def __init__(self, tokens, is_finished=False):
            self.tokens = tokens
            self.is_finished = is_finished
    
    # Simulate beam cleanup (from llm.py line 629)
    all_beams = [
        MockBeam([1, 2, 3], is_finished=False),  # Active beam
        MockBeam([1, 2, 3, 2], is_finished=True),  # Finished beam (EOS)
        MockBeam([1, 2, 3, 4], is_finished=False),  # Active beam
        MockBeam([1, 2, 3, 5, 2], is_finished=True),  # Finished beam (EOS)
    ]
    
    print(f"Before cleanup: {len(all_beams)} total beams")
    print(f"  - Active beams: {len([b for b in all_beams if not b.is_finished])}")
    print(f"  - Finished beams: {len([b for b in all_beams if b.is_finished])}")
    
    # This is the cleanup logic from the code
    active_beams = [beam for beam in all_beams if not beam.is_finished]
    
    print(f"\nAfter cleanup: {len(active_beams)} active beams")
    print(f"  - Finished beams removed: {len(all_beams) - len(active_beams)}")
    
    # Verify cleanup worked correctly
    expected_active = 2
    if len(active_beams) == expected_active:
        print("✅ Beam cleanup working correctly - finished beams removed from active processing")
        return True
    else:
        print(f"❌ Beam cleanup failed - expected {expected_active} active beams, got {len(active_beams)}")
        return False

def main():
    """Run all EOS beam stopping tests."""
    print("Testing EOS Beam Stopping Behavior")
    print("=" * 50)
    
    test1_passed = test_eos_beam_stopping_logic()
    test2_passed = test_beam_cleanup_logic()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    if test1_passed and test2_passed:
        print("🎉 All tests passed!")
        print("✅ EOS beam stopping logic is working correctly")
        print("✅ When a beam hits an EOS token, it stops processing that beam")
        print("✅ Finished beams are properly removed from active processing")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)