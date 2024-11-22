import argparse
import ImageAnalysisModel as pa

def main(image_folder_path, containerWidth):
    if not image_folder_path or not containerWidth:
        print("Image path and container width must be provided.")
        return

    checkpoint_folder = '/checkpoints'

    analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth)
    analyser.evenLighting()
    analyser.overlayImage()
    analyser.showImage()
    Testing = False
    analyser.analyseParticles(checkpoint_folder, Testing)
    analyser.showMasks()
    bins = [38, 106, 1000, 8000]  # bins
    analyser.saveResults(bins)
    analyser.formatResults()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run particle analysis on provided image folder.")
    parser.add_argument('image_folder_path', type=str, help='Path to the folder containing sample images.')
    parser.add_argument('containerWidth', type=str, help='Container width in um for example 18(cm) is equal to 180000(um).')

    args = parser.parse_args()

    main(args.image_folder_path, args.containerWidth)
