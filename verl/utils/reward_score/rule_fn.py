from typing import List, Dict, Any


def math_verify(model_response: str, parameters: Dict[str, Any]) -> bool:
    """
    Verify model response using mathematical verification logic
    Priority: math_verify.py first, then fallback to dapo verification if failed
    
    Args:
        model_response: Model's response
        parameters: Verification parameter dictionary, format: {"answer": "answer"}
        
    Returns:
        bool: Verification result, True indicates correct, False indicates incorrect
    """
    
    from verl.utils.reward_score.math_verify import compute_score as compute_score_verify
    from verl.utils.reward_score.math_dapo import compute_score as compute_score_dapo
    
    # Extract answer from dictionary
    answer = parameters.get("answer")
    if answer is None:
        raise ValueError("math_verify: parameters dict must contain 'answer' key")
    
    # Priority 1: Use math_verify.py verification
    math_verify_passed = False
    try:
        score_verify = compute_score_verify(model_response, answer)
        if score_verify > 0.5:
            math_verify_passed = True
            return True
    except Exception as e:
        print(f"math_verify: Verify verification failed with error: {e}")
    
    # Priority 2: Fallback to dapo verification if verify verification failed
    try:
        result_dapo = compute_score_dapo(model_response, answer, strict_box_verify=True)
        acc_dapo = result_dapo.get("acc", False)
        
        # If math_verify failed but dapo passed, print the response and answer
        if not math_verify_passed and acc_dapo:
            print(f"math_verify: math_verify failed but dapo passed")
            print(f"  Response: {model_response}")
            print(f"  Answer: {answer}")
        
        return acc_dapo
    except Exception as e:
        print(f"math_dapo: Dapo verification failed with error: {e}")
        return False


def word_count_range(model_response: str, parameters: Dict[str, Any]) -> bool:
    """
    Verify if the word count of model response is within the specified range
    
    Args:
        model_response: Model's response
        parameters: Verification parameter dictionary, format: {"min_count": 10, "max_count": 100}
        
    Returns:
        bool: Verification result, True indicates word count is within range, False indicates out of range
        
    Raises:
        ValueError: When parameter format is incorrect
    """
    # Extract min_count and max_count from dictionary
    min_count = parameters.get("min_count")
    max_count = parameters.get("max_count")
    
    if min_count is None or max_count is None:
        raise ValueError("word_count_range: parameters dict must contain 'min_count' and 'max_count' keys")
    
    try:
        min_count = int(min_count)
        max_count = int(max_count)
    except (ValueError, TypeError):
        raise ValueError(f"word_count_range: min_count and max_count must be integers, got min_count={min_count}, max_count={max_count}")
    
    if min_count > max_count:
        raise ValueError(f"word_count_range: min_count ({min_count}) cannot be greater than max_count ({max_count})")
        
    word_count = len(model_response.split())
    
    result = min_count <= word_count <= max_count
    print(f"word_count_range: response has {word_count} words, range [{min_count}, {max_count}], result: {result}")
    return result


# Verification function registry for dynamic invocation
VERIFICATION_FUNCTIONS = {
    'math_verify': math_verify,
    'word_count_range': word_count_range,
}


def get_verification_function(function_name: str):
    """
    Get verification function by function name
    
    Args:
        function_name: Verification function name
        
    Returns:
        Verification function or None (if function does not exist)
    """
    return VERIFICATION_FUNCTIONS.get(function_name)