# import inspect
# import builtins

# def print(*args, **kwargs):
#     # Get the caller's frame and extract the function name
#     caller_frame = inspect.currentframe().f_back
#     caller_function = inspect.getframeinfo(caller_frame).function
    
#     # Add the caller information to the output
#     caller_info = f"Called from function: {caller_function}"
#     args = (*args, caller_info)
    
#     # Call the original print function from the builtins module to perform the actual printing
#     builtins.print(*args, **kwargs)
# def another():
#     print("Hello, world!")

# # Test the overridden print function
# def example_function():
#     print("Hello, world!")
#     another()

# example_function()

import inspect
import builtins

def print(*args, **kwargs):
    # Get the caller's frame and extract the function name
    caller_frame = inspect.currentframe().f_back
    caller_function = inspect.getframeinfo(caller_frame).function
    
    # Get the call stack
    call_stack = []
    frame = caller_frame
    while frame is not None:
        function_name = inspect.getframeinfo(frame).function
        call_stack.append(function_name)
        frame = frame.f_back
    
    # Reverse the call stack to print the functions in the correct order
    call_stack.reverse()
    
    # Format the call stack as a string
    call_stack_info = "Call Stack:\n" + " -> ".join(call_stack)
    
    # Add the call stack information to the output
    args = (*args, call_stack_info)
    
    # Call the original print function from the builtins module to perform the actual printing
    builtins.print(*args, **kwargs)

# Test the overridden print function
def function1():
    function2()

def function2():
    for i in range(5):
        print("Hello, world!")

function1()
