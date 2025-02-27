"""
charon_detector
===============

Core YOLO detection module that recursively processes images in a given
directory (grouped by treatment subfolders), applies YOLOv11 object detection,
and outputs structured results (Excel or CSV) along with optional annotated images.

.. module:: charon_detector
   :synopsis: Core object detection logic for YOLOv11 pipeline.
"""

import os
import cv2
import logging
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict

CLASS_LABELS = ["dead", "alive", "ginko"]

logger = logging.getLogger(__name__)

def run_detection(directory, get_result_images=False, output_format='xlsx'):
    """
    Recursively run object detection on all images in ``directory`` using a YOLOv11 model.

    This function:
    
    1. Recursively discovers all images (JPG, PNG, etc.) in ``directory``.
    2. Groups images by treatment based on the first-level subfolder name.
    3. Loads the YOLO model from ``weights/best.pt``.
    4. For each image:
       - Applies YOLO to detect objects.
       - Maps numeric classes (0,1,2) to human-readable labels in ``CLASS_LABELS``.
       - Optionally draws bounding boxes on the image and saves them under an
         ``annotated/`` folder (if ``get_result_images=True``).
    5. Produces a multi-sheet XLSX file or multiple CSV files summarizing:
       - Overall counts of each class per treatment in a "Summary" (sheet or CSV).
       - Detailed bounding box detections per treatment.

    :param directory: Path to the parent directory containing subfolders of images.
    :type directory: str

    :param get_result_images: If True, annotated images are saved in a mirrored
                              ``annotated/`` folder preserving subfolder structure.
                              Defaults to False.
    :type get_result_images: bool, optional

    :param output_format: Either ``'xlsx'`` or ``'csv'``. Defaults to ``'xlsx'``.
                          If ``'xlsx'``, the script generates a single Excel file
                          with multiple sheets. If ``'csv'``, it generates a
                          folder of CSVs (one CSV per treatment and one summary CSV).
    :type output_format: str, optional

    :raises FileNotFoundError: If ``directory`` does not exist or if no images are found.
    :raises Exception: If the YOLO model fails to load or critical I/O errors occur.

    :returns: None
    :rtype: None

    Usage Example (programmatically)::

        from charon_detector import run_detection

        # Run detection on 'my_data' folder, save annotated images, export XLSX
        run_detection('my_data', get_result_images=True, output_format='xlsx')

    .. note::
       The summary sheet or file lists each treatment and the total counts of
       "dead", "alive", and "ginko". Additional sheets or CSV files detail each
       bounding box detection with columns:
       ``[imagename, class, confidence, xmin, ymin, xmax, ymax]``.

    .. warning::
       The current YOLO weights (best.pt, last.pt) may internally label classes
       as (0=alive, 1=dead, 2=ginko). This script overrides them so that the
       final output is consistently (0=dead, 1=alive, 2=ginko). In the future,
       retraining the model with the correct labeling will remove this mismatch.

    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return

    # Recursively find images
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_paths = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, fname))
    if not image_paths:
        logger.error(f"No images found in directory: {directory}")
        return

    # Prepare place to store detection data:
    # summary_counts[treatment] = {"dead": ..., "alive": ..., "ginko": ...}
    summary_counts = defaultdict(lambda: {"dead": 0, "alive": 0, "ginko": 0})
    # details[treatment] = list of dicts { "imagename": ..., "class": ..., "confidence": ..., "xmin":..., ...}
    details = defaultdict(list)

    # If saving annotated images, create a base 'annotated' directory
    annotated_base = None
    if get_result_images:
        annotated_base = os.path.join(directory, "annotated")
        try:
            os.makedirs(annotated_base, exist_ok=True)
            logger.info(f"Annotated images will be saved to '{annotated_base}'")
        except Exception as e:
            logger.error(f"Could not create annotated output directory: {e}")
            return

    # Load YOLO model from weights
    weights_path = os.path.join("weights", "best.pt")
    logger.info(f"Loading YOLOv11 model from '{weights_path}'...")
    try:
        model = YOLO(weights_path)  # Ultralytics YOLO
    except Exception as e:
        logger.error(f"Failed to load model from {weights_path}: {e}")
        return
    logger.info("Model loaded successfully")

    # Process each image
    total_images = len(image_paths)
    for idx, img_path in enumerate(sorted(image_paths), start=1):
        rel_path = os.path.relpath(img_path, directory)
        logger.info(f"[{idx}/{total_images}] Processing '{rel_path}'")

        # Derive treatment name from immediate subfolder under directory
        # e.g. parent_dir/treatment_hypoxia/image_001.png -> "treatment_hypoxia"
        # or if parent_dir/itself.png -> "parent_dir" fallback
        parts = rel_path.split(os.sep)
        # If the image is in a subfolder, we use the first part of the path as treatment
        if len(parts) > 1:
            treatment = parts[0]
        else:
            # If the image is directly under parent_dir, treat the parent dir name as 'treatment'
            treatment = os.path.basename(directory)

        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            logger.error(f"Failed to read image '{img_path}'")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run inference
        try:
            results = model.predict(source=img_rgb, verbose=False)
        except Exception as e:
            logger.error(f"Inference error for '{rel_path}': {e}")
            continue

        if not results:
            logger.warning(f"No result returned for '{rel_path}'")
            continue
        result = results[0]

        boxes = result.boxes
        detection_count = 0

        h, w = img_bgr.shape[:2]

        if boxes is not None:
            for box in boxes:
                detection_count += 1
                cls_val = box.cls
                conf_val = box.conf
                cls_id = int(cls_val[0]) if hasattr(cls_val, "__len__") else int(cls_val)
                conf = float(conf_val[0]) if hasattr(conf_val, "__len__") else float(conf_val)

                # Map class ID -> label
                if 0 <= cls_id < len(CLASS_LABELS):
                    label = CLASS_LABELS[cls_id]
                else:
                    label = f"unknown_{cls_id}"

                # bounding box
                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, coords)
                # clamp
                x1, x2 = max(x1, 0), min(x2, w-1)
                y1, y2 = max(y1, 0), min(y2, h-1)

                # Update summary counts if it's one of our known categories
                if label in summary_counts[treatment]:
                    summary_counts[treatment][label] += 1

                # Save detail row
                details[treatment].append({
                    "imagename": os.path.basename(img_path),
                    "class": label,
                    "confidence": round(conf, 4),
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2
                })

                # Draw bounding box if requested
                if get_result_images:
                    if label == "dead":
                        color = (0, 0, 255)   # BGR = red
                    elif label == "alive":
                        color = (0, 255, 0)  # green
                    else:
                        color = (255, 0, 0)  # blue
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                    text = f"{label} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    tx, ty = x1, y1 - 5
                    if ty - th < 0:
                        ty = y1 + th + 5
                    # Draw background for text
                    cv2.rectangle(img_bgr, (tx, ty - th - 2), (tx + tw, ty), color, cv2.FILLED)
                    cv2.putText(img_bgr, text, (tx, ty - 2), font, font_scale,
                                (255, 255, 255), thickness, cv2.LINE_AA)

            logger.info(f"Detected {detection_count} object(s) in '{rel_path}'")

        # Save annotated image if requested
        if get_result_images:
            # mirrored path under 'annotated' base
            ann_path = os.path.join(annotated_base, rel_path)
            ann_dir = os.path.dirname(ann_path)
            os.makedirs(ann_dir, exist_ok=True)
            cv2.imwrite(ann_path, img_bgr)

    # Create the multi-sheet Excel / CSV structure
    # We'll create a single Excel with:
    #   - Sheet1: 'Summary' = treatment, alive, dead, ginko
    #   - For each treatment: a sheet with columns [imagename, class, confidence, xmin, ymin, xmax, ymax]

    if output_format.lower() == "xlsx":
        output_xlsx = os.path.join(directory, "detection_results.xlsx")
        try:
            with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
                # Build summary DataFrame
                # summary_counts[treatment] = { 'dead': x, 'alive': y, 'ginko': z }
                summary_rows = []
                for treatment, cat_dict in summary_counts.items():
                    summary_rows.append({
                        "Treatment": treatment,
                        "dead": cat_dict["dead"],
                        "alive": cat_dict["alive"],
                        "ginko": cat_dict["ginko"]
                    })
                df_summary = pd.DataFrame(summary_rows, columns=["Treatment", "alive", "dead", "ginko"])
                # Reorder columns if you want (the user wants: Treatment, alive, dead, ginko)
                df_summary.to_excel(writer, sheet_name="Summary", index=False)

                # Per-treatment sheets
                for treatment, records in details.items():
                    df_det = pd.DataFrame(records, columns=[
                        "imagename", "class", "confidence", "xmin", "ymin", "xmax", "ymax"
                    ])
                    # Sort by imagename to keep it tidy
                    df_det.sort_values("imagename", inplace=True)
                    sheet_name = str(treatment)[:31]  # excel limit
                    df_det.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Saved multi-sheet results to '{output_xlsx}'")
        except Exception as e:
            logger.error(f"Failed to write XLSX file: {e}")

    else:  # CSV output
        # We'll mimic the multi-sheet approach by creating a folder with separate CSV files.
        out_folder = os.path.join(directory, "detection_results")
        os.makedirs(out_folder, exist_ok=True)

        # 1) Summary.csv
        summary_rows = []
        for treatment, cat_dict in summary_counts.items():
            summary_rows.append({
                "Treatment": treatment,
                "alive": cat_dict["alive"],
                "dead": cat_dict["dead"],
                "ginko": cat_dict["ginko"]
            })
        df_summary = pd.DataFrame(summary_rows, columns=["Treatment", "alive", "dead", "ginko"])
        df_summary.to_csv(os.path.join(out_folder, "Summary.csv"), index=False)

        # 2) One CSV per treatment
        for treatment, records in details.items():
            df_det = pd.DataFrame(records, columns=[
                "imagename", "class", "confidence", "xmin", "ymin", "xmax", "ymax"
            ])
            # Sort by imagename
            df_det.sort_values("imagename", inplace=True)
            # Safe filename
            safe_name = treatment.replace("/", "_")  # if there's a slash
            csv_path = os.path.join(out_folder, f"{safe_name}.csv")
            df_det.to_csv(csv_path, index=False)

        logger.info(f"Saved CSV results in '{out_folder}'")
