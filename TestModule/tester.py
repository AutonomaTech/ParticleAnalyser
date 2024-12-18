import ImageAnalysisModel as pa
import os

current_path = os.getcwd()

# Construct the absolute paths by joining the current working directory with relative paths
image_folder_path = os.path.join(current_path, "Samples/RCB1763013_S1")
checkpoint_folder = os.path.join(current_path, "checkpoints")

# in um
containerWidth = 180000
# initialise analyser
analyser = pa.ImageAnalysisModel(image_folder_path, containerWidth=containerWidth,config_path='config.ini')

# Post processing of the image
analyser.color_correction()
analyser.evenLighting()
analyser.overlayImage()


# analyser.showImage()
# analyser.crop_image()
# industry standard
# bins: 0.038, 0.106, 1, 8 (mm)--INDUSTRY STANDARD
bins = [38, 106, 1000, 8000]

# bins = [1000, 2000, 3000, 4000, 5000, 7000, 8000]
# bins = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # sample 0002


"""
analyser.setBins(bins)
#ONLY WITH GPU
analyser.analyseParticles(checkpoint_folder,False)
analyser.showMasks()
analyser.saveResults()
"""
analyser.loadSegments(checkpoint_folder, bins)
# analyser.analysesCV2()
# analyser.setScalingFactor(1)
analyser.savePsdData()
analyser.savePsdDataWithDiameter()
analyser.formatResults(True)
analyser.plotBins()