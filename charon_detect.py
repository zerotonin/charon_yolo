"""
charon_detect
=============

A CLI wrapper for charon_detector.py. It parses command-line arguments and invokes
the main detection function `run_detection`. Designed to be run as a script.

.. module:: charon_detect
   :synopsis: Command-line interface for YOLOv11 detection pipeline.
"""

import argparse
import logging
from charon_detector import run_detection

def main():
    """
    Main command-line entry point.

    This function:

    1. Parses command-line arguments to determine:
       - The target directory containing subfolders of images.
       - Whether annotated images should be saved.
       - Which output format to use (XLSX or CSV).

    2. Configures logging to display informational messages.

    3. Calls the `run_detection` function from `charon_detector` to perform
       the actual YOLO object detection workflow.

    Example command-line usage::

        python charon_detect.py /path/to/parent_dir --get_resultimages --output xlsx

    .. note::
       If you are running this script outside of a properly configured environment,
       ensure that dependencies (Ultralytics, OpenCV, pandas, etc.) are installed.

    :raises SystemExit: If the user provided invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect objects in subfolders using YOLOv11 with multi-sheet XLSX or CSV output."
    )
    parser.add_argument("directory", help="Path to the parent directory with treatment subfolders.")
    parser.add_argument("--get_resultimages", action="store_true",
                        help="If set, saves annotated images in a mirrored 'annotated/' folder.")
    parser.add_argument("--output", choices=["xlsx", "csv"], default="xlsx",
                        help="Output format (xlsx for multi-sheet or csv for multiple CSVs).")
    args = parser.parse_args()

    # Optionally configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    run_detection(
        directory=args.directory,
        get_result_images=args.get_resultimages,
        output_format=args.output
    )

if __name__ == "__main__":
    main()
