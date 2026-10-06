import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Model identifier
DEFAULT_IMAGE_MODEL = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
CONFIDENCE_THRESHOLD = 0.85

_classifier_instance = None


def get_image_classifier():
    """Lazily loads the plant disease identification pipeline into memory."""
    global _classifier_instance
    if _classifier_instance is None:
        try:
            from transformers import pipeline
            logger.info(f"Loading plant disease classifier '{DEFAULT_IMAGE_MODEL}'...")
            _classifier_instance = pipeline(
                "image-classification",
                model=DEFAULT_IMAGE_MODEL
            )
            logger.info("Plant disease image classifier successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load image classification pipeline: {e}")
            raise
    return _classifier_instance


def diagnose_plant_image(image_path: str, threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """
    Runs inference on an agricultural leaf image and enforces confidence threshold.
    
    Returns:
        Dict with keys 'success' (bool), 'disease' (str), 'confidence' (float), and 'message' (str).
    """
    if not os.path.exists(image_path):
        return {
            "success": False,
            "message": f"Image file not found at path: {image_path}"
        }

    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image file '{image_path}': {e}")
        return {
            "success": False,
            "message": "Invalid or unreadable image file. Please upload a clear JPG/PNG photo."
        }

    try:
        classifier = get_image_classifier()
        predictions = classifier(img)
    except Exception as e:
        logger.error(f"Inference error in plant disease classifier: {e}")
        return {
            "success": False,
            "message": f"Unable to run plant diagnosis model: {str(e)}"
        }

    if not predictions:
        return {
            "success": False,
            "message": "No classification predictions could be generated for this image."
        }

    top_prediction = predictions[0]
    label = top_prediction.get("label", "Unknown Disease")
    score = float(top_prediction.get("score", 0.0))

    if score < threshold:
        return {
            "success": False,
            "disease": label,
            "confidence": score,
            "message": (
                f"I cannot clearly identify the issue with high confidence (Top guess: {label} at {score:.1%}). "
                "Please provide a closer, clearer shot of the affected leaf or crop against a plain background."
            )
        }

    return {
        "success": True,
        "disease": label,
        "confidence": score,
        "message": f"Identified {label} with {score:.1%} confidence."
    }