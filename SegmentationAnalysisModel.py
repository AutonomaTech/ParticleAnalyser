import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from datetime import datetime
import sys
import ParticleSegmentationModel as psa
import ImageProcessingModel as ip
import logger_config

#industry standard
bins=[0, 38, 106, 1000, 8000] #bins: 0.038, 0.106, 1, 8 (mm)--INDUSTRY STANDARD

class SegmentationAnalysisModel:
    def __init__(self, image_folder_path,sampleID ,containerWidth):

        #initialise image
        self.sampleID=sampleID
        self.imageProcessor=ip.ImageProcessingModel(image_folder_path,sampleID)  
        #overlay the image before processing it
        self.imageProcessor.overlayImage()
        self.imagePath=self.imageProcessor.getImagePath()

        #Scaling factor needs to be reviewed, as it currently only works with cropped images 
            #that do not account for the container's borders
        #width of bucket(mm)/ width of bucket in pixels(W_image_pixels)
        W_image_pixels=self.imageProcessor.getWidth()
        scalingFactor = containerWidth / W_image_pixels
        self.setScalingFactor(scalingFactor)
        self.setScalingNumber(W_image_pixels)
        self.scalingStamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        print(self.scalingStamp)
        
        self.analysisTime=0
        self.numberofBins=0

        self.p=None
        
    #set scaling factor
    def setScalingFactor(self, scalingFactor):
        self.scalingFactor=scalingFactor

    #visualise mask, so output what the results are from the first image
    def visualiseMasks(self):
        self.p.visualiseMasks()
    
    def showImage(self):
      self.imageProcessor.showImage()

    def setScalingNumber(self,scalingNumber):
      self.scalingNumber=scalingNumber
      
    def showMasks(self):
        self.p.visualise_masks()

    # set a difenrt number of bins based on the softwrae requirements
    #set the number fo bins, call if it needs to be changed from 4
    def setBins(self, bins):
        self.numberofBins=len(bins)
        self.p.bins=bins

    #analyse the article s in the image with the same model
    def analyseParticles(self,checkpoint_folder):
        def loadSamModel(checkpoint_folder):
            os.makedirs(checkpoint_folder, exist_ok=True)
            #checkpoint_filename = "sam_vit_h_4b8939.pth" #SAM
            checkpoint_filename = "sam2.1_hiera_large.pt" #SAM2
            CHECKPOINT_PATH = os.path.join(checkpoint_folder, checkpoint_filename)

            return CHECKPOINT_PATH
        #calculate how long it takes
        def calculateAnalysisTime(startTime,endTime):
            duration = endTime - startTime
            total_seconds = duration.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            self.analysisTime = f"PT{minutes}M{seconds:.1f}S"
        
        CHECKPOINT_PATH=loadSamModel(checkpoint_folder)
        startTime=datetime.now()
        self.p = psa.ParticleSegmentationModel(self.imagePath, \
                                     CHECKPOINT_PATH ,
                                     self.scalingFactor
                                     )
       
        self.p.generate_mask()

        endTime=datetime.now()
        calculateAnalysisTime(startTime,endTime)

    
    # save :
    #SampleID.csv 
    #SampleID_distribution.txt
    def saveResults(self):
        folder_path=self.imageProcessor.getImageFolder()
        csv_filename = os.path.join(folder_path, f"{self.sampleID}.csv")

        self.p.setdiameter_threshold(10)

        self.p.save_masks_to_csv(csv_filename)
        print(f"--> Masks saved to CSV file: {csv_filename}")

        if self.p.bins is None:
            self.numberofBins = len(bins)  # Assuming bins are defined elsewhere with a length of 5
            self.p.bins = bins

        # txt_filename = os.path.join(self.image_folder_path, f"{self.sampleID}_segments.txt")
        # self.p.save_segments(txt_filename)
        self.p.get_psd_data()
        self.p.save_psd_as_csv(self.sampleID, folder_path)
        print(f"--> PSD data saved as CSV in folder: {folder_path}")

    def formatResults(self):
        self.totArea=self.p.get_totalArea()
        print("-----------------------------------------------")
        print("Sample ID:", self.sampleID)
        print("Total Area:", self.totArea)
        print("Scaling Factor:", self.scalingFactor)
        print("Scaling Number:", self.scalingNumber)
        print("Intensity:", self.imageProcessor.getIntensity())
        print("Scaling Stamp:", self.scalingStamp)
        print("Analysis Time:", self.analysisTime)
        print("Number of Particles:", self.numberofBins)
        print("-----------------------------------------------")

        #TODO --li
        #sizeAnalysisModel(self.sampleID,self.totArea,self.scalingFactor,self.scalingNumber,self.intensity, \
                       # self.scalingStamp,self.analysisTime, self.numberofBins )

    
    
    





