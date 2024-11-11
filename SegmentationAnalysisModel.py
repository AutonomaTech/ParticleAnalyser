import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from datetime import datetime
import sys
import ParticleSegmentationModel as psa
import logger_config

#industry standard
bins=[0, 38, 106, 1000, 8000] #bins: 0.038, 0.106, 1, 8 (mm)--INDUSTRY STANDARD

class SegmentationAnalysisModel:
    def __init__(self, imagePath,scalingFactor):
        self.imagePath=imagePath

        self.scalingStamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        print(self.scalingStamp)
        self.setScalingFactor(scalingFactor)
        
        self.intensity=0.0
        self.scalingNumber=1
        self.analysisTime=0
        self.numberofBins=0

        self.p=None
    #set scaling factor
    def setScalingFactor(self, scalingFactor):
        self.scalingFactor=scalingFactor

    #visualise mask, so output what the results are from the first image
    def visualiseMasks(self):
        self.p.visualiseMasks()

    # set a difenrt number of bins based on the softwrae requirements
    #set the number fo bins, call if it needs to be changed from 4
    def setBins(self, bins):
        self.numberofBins=len(bins)
        self.p.bins=bins

    #analyse the article s in the image with the same model
    def analyseParticles(self):
        def loadSamModel():
            weights_dir = "/content/weights"
            os.makedirs(weights_dir, exist_ok=True)
            checkpoint_filename = "sam_vit_h_4b8939.pth"
            CHECKPOINT_PATH = os.path.join(weights_dir, checkpoint_filename)

            return CHECKPOINT_PATH
        #calculate how long it takes
        def calculateAnalysisTime(startTime,endTime):
            duration = endTime - startTime
            total_seconds = duration.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            self.analysisTime = f"PT{minutes}M{seconds:.1f}S"
        
        CHECKPOINT_PATH=loadSamModel()

        startTime=datetime.now()
        self.p = psa.ParticleSegmentationModel(self.imagePath, \
                                     CHECKPOINT_PATH ,
                                     self.scalingFactor
                                     )
       
        self.p.generate_mask()

        endTime=datetime.now()
        calculateAnalysisTime(startTime,endTime)

    def showMasks(self):
        self.p.visualise_masks()
    # save :
    #SampleID.csv 
    #SampleID_segments.txt
    #SampleID_distribution.txt
    def saveResults(self, image_folder_path):
        folder_path, file_name_with_ext = os.path.split(image_folder_path)
        file_name, _ = os.path.splitext(file_name_with_ext)
        self.sampleID = file_name

        csv_filename = os.path.join(folder_path, f"{self.sampleID}.csv")
        
        self.p.setdiameter_threshold(10)

        self.p.save_masks_to_csv(csv_filename)
        print(f"Masks saved to CSV file: {csv_filename}")

        if self.p.bins is None:
            self.numberofBins = len(bins)  # Assuming bins are defined elsewhere with a length of 5
            self.p.bins = bins

        # txt_filename = os.path.join(self.image_folder_path, f"{self.sampleID}_segments.txt")
        # self.p.save_segments(txt_filename)
        self.p.get_psd_data()
        self.p.save_psd_as_csv(self.sampleID, folder_path)
        print(f"PSD data saved as CSV in folder: {folder_path}")

       
    def formatResults(self):
        self.totArea=self.p.get_totalArea()

        print("Sample ID:", self.sampleID)
        print("Total Area:", self.totArea)
        print("Scaling Factor:", self.scalingFactor)
        print("Scaling Number:", self.scalingNumber)
        print("Intensity:", self.intensity)
        print("Scaling Stamp:", self.scalingStamp)
        print("Analysis Time:", self.analysisTime)
        print("Number of Particles:", self.numberofBins)

        #TODO --li
        #sizeAnalysisModel(self.sampleID,self.totArea,self.scalingFactor,self.scalingNumber,self.intensity, \
                       # self.scalingStamp,self.analysisTime, self.numberofBins )

    
    
    





