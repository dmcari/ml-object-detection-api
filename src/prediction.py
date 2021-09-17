import cvlib as cv
from cvlib.object_detection import draw_bbox


def detect_and_draw_box(image, model="yolov3-tiny", confidence=0.5):
    """Detects common objects on an image and creates a new image
        with bounding boxes.

    Args:
        image (array): Image as a numpy array.
        model (str): Either "yolov3" or "yolov3-tiny".
            Defaults to "yolov3-tiny".
        confidence (float, optional): Desired confidence level.
            Defaults to 0.5.
    """

    # Perform the object detection
    bbox, label, conf = cv.detect_common_objects(
        image, confidence=confidence, model=model
    )

    # Print detected objects with confidence level
    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")

    # Return a new image that includes the bounding boxes
    return draw_bbox(image, bbox, label, conf, write_conf=True)
