"""
This script uses monkey-patching to robustly inject a BboxProcessorStep
into the lerobot_record.py script's processing pipeline.

It first calls the original processor factory to get the policy-specific
pipeline, then checks if the custom step is already present, and if not,
injects it at the beginning.
"""
import logging

# Import the original script and the specific function we want to patch
import lerobot.scripts.lerobot_record as original_record_script
from lerobot.policies.factory import make_pre_post_processors as original_make_processors

# Import the components needed for the check and injection
from lerobot.processor.bbox_processor import BboxProcessorStep


def custom_make_processors_with_bbox_injection(*args, **kwargs):
    """
    A wrapper that gets the original preprocessor pipeline and injects
    the BboxProcessorStep at the beginning if it's not already there.
    """
    logging.info("[MONKEY-PATCH] Intercepted call to make_pre_post_processors.")

    # 1. Call the original function to get the correct, policy-specific pipeline.
    preprocessor, postprocessor = original_make_processors(*args, **kwargs)

    # 2. Check if a BboxProcessorStep is already in the pipeline.
    already_has_bbox = any(isinstance(step, BboxProcessorStep) for step in preprocessor.steps)

    # 3. If not, inject it at the beginning of the steps list.
    if not already_has_bbox:
        logging.info("[MONKEY-PATCH] BboxProcessorStep not found. Injecting it at the beginning.")
        
        # NOTE: This path is hardcoded for this example. You may need to make this
        # configurable if you have different bounding box processor configs.
        bbox_config_path = "configs/custom_processors/bbox_preprocessor_cfg.json"
        logging.info(f"[MONKEY-PATCH] Loading BboxProcessorStep from: {bbox_config_path}")
        
        bbox_step = BboxProcessorStep.from_json(bbox_config_path)

        # By injecting the step first, we add new observation data (e.g., bounding boxes)
        # before any other processing happens.        
        preprocessor.steps.insert(0, bbox_step)
        
        logging.info("[MONKEY-PATCH] Injection successful. New preprocessor steps:")
        for i, step in enumerate(preprocessor.steps):
            logging.info(f"  {i}: {step.__class__.__name__}")
            
    else:
        logging.info("[MONKEY-PATCH] BboxProcessorStep already present. Skipping injection.")

    # 4. Return the modified preprocessor and original postprocessor.
    return preprocessor, postprocessor


def main():
    """
    Applies the monkey-patch and runs the original `lerobot_record` script.
    """
    # --- THE MONKEY PATCH ---
    # Replace the function in the original module with our custom version.
    # This must be done BEFORE the original main() is called.
    original_record_script.make_pre_post_processors = custom_make_processors_with_bbox_injection

    # Now, run the original main function. It will unknowingly use our patched function.
    print("--- Starting modified lerobot-record script ---")
    print("BboxProcessorStep will be injected into the pipeline.")
    original_record_script.main()
    print("--- Modified lerobot-record script finished ---")


if __name__ == "__main__":
    main()