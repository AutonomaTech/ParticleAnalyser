
import argparse
import ImageAnalysisModel as pa
import traceback
from logger_config import get_logger
logger = get_logger("StartUp")
def main(image_folder_path, containerWidth,config_path):
    """
    Main function to execute the image analysis process.
    """
    try:
        logger.info(f"Starting analysis with folder: {image_folder_path}, container width: {containerWidth}")
        analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth=containerWidth, config_path=config_path)
        analyser.run_analysis()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    # Parse command-line arguments
    """
     Main function to execute the image analysis process.
    """
    parser = argparse.ArgumentParser(description="Run particle analysis on the provided image folder.")
    parser.add_argument('image_folder_path', type=str, help='Path to the folder containing sample images.')
    parser.add_argument('--containerWidth', default=180000,type=int,help='Container width in micrometers, for example, 18 cm is equal to 180000 µm.')
    parser.add_argument('--config_path', type=str, default='config.ini',
                        help='config file path')
    args = parser.parse_args()

    main(args.image_folder_path,args.containerWidth,args.config_path)


