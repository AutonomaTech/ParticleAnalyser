
import ImageAnalysisModel as pa

def main(image_folder_path,containerWidth):
    # Check if the image_folder_path or containerWidth are empty or None
    if not image_folder_path or not containerWidth:
        print("Error: Image folder path and container width must be specified.")
        return
    checkpoint_folder = '/checkpoints'
    # in um

    # initialise analyser
    analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth)
    analyser.evenLighting()
    analyser.overlayImage()
    # analyser.crop_image() # cannot do this in colab
    analyser.showImage()
    Testing = False
    analyser.analyseParticles(checkpoint_folder, Testing)
    analyser.showMasks()
    bins = [0, 38, 106, 1000, 8000]  # bins: bottom 0.038, 0.106, 1, 8 (mm)--INDUSTRY STANDARD
    # analyser.setBins(bins)
    analyser.saveResults(bins)
    analyser.formatResults()

if __name__ == "__main__":
    main()
