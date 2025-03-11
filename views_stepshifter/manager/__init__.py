from multiprocessing import set_start_method, get_start_method
from views_stepshifter.manager.stepshifter_manager import StepshifterManager

try:
    set_start_method('spawn')
    print(f"Multiprocessing start method set to: {get_start_method()}") # Use print because logger is not yet configured
except RuntimeError:
    # print(f"Multiprocessing start method is already set to: {get_start_method()}") 
    pass

__all__ = ["StepshifterManager"]