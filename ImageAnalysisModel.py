import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from datetime import datetime
import sys
import ParticleSegmentationModel as psa
import ImageProcessingModel as ip
import logger_config
import sizeAnalysisModel as sa
import ContainerScalerModel as cs


class ImageAnalysisModel:
    def __init__(self, image_folder_path,sampleID ,containerWidth):

        #initialise image
        self.sampleID=sampleID
        self.imageProcessor=ip.ImageProcessingModel(image_folder_path,sampleID)  
        #overlay the image before processing it
        self.imageProcessor.overlayImage()
        self.imagePath=self.imageProcessor.getImagePath()

        self.ContainerScaler=cs.ContainerScalerModel(containerWidth,self.imageProcessor.getWidth())

        self.analysisTime=0
        self.p=None
        

    #visualise mask, so output what the results are from the first image
    def visualiseMasks(self):
        self.p.visualiseMasks()
    
    def showImage(self):
      self.imageProcessor.showImage()

    def showMasks(self):
        self.p.visualise_masks()

    # set a difenrt number of bins based on the softwrae requirements
    #set the number fo bins, call if it needs to be changed from 4
    def setBins(self, bins):
        if self.p is not None:
            self.numberofBins = len(bins)
            self.p.bins = bins

    def loadModel(self,checkpoint_folder):
        def loadSamModel(checkpoint_folder):
            os.makedirs(checkpoint_folder, exist_ok=True)
            #checkpoint_filename = "sam_vit_h_4b8939.pth" #SAM
            checkpoint_filename = "sam2.1_hiera_large.pt" #SAM2
            CHECKPOINT_PATH = os.path.join(checkpoint_folder, checkpoint_filename)

            return CHECKPOINT_PATH
        CHECKPOINT_PATH=loadSamModel(checkpoint_folder)
        self.p = psa.ParticleSegmentationModel(self.imagePath, \
                                     CHECKPOINT_PATH ,
                                     self.ContainerScaler.scalingFactor
                                     )
    #analyse the articles in the image with the same model
    def analyseParticles(self,checkpoint_folder,testing):
        
        #calculate how long it takes
        def calculateAnalysisTime(duration):
            total_seconds = duration.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            self.analysisTime = f"PT{minutes}M{seconds:.1f}S"
        
        self.loadModel(checkpoint_folder)
        if testing:
          self.p.testing_generate_mask()
        else:  
          self.p.generate_mask()
        
        calculateAnalysisTime(self.p.getExecutionTime()) 
    # save :
    #SampleID.csv 
    #SampleID_distribution.txt
    def savePsdData(self):
        self.p.get_psd_data()
        self.distributions_filename = os.path.join(self.folder_path, f"{self.sampleID}_distribution.txt")
        self.p.save_psd_as_txt(self.sampleID, self.folder_path)
        print(f"--> PSD data saved as TXT file: {self.distributions_filename}")

    def saveResults(self):
        if self.p.bins is None:
            raise ValueError("Bins are not defined. Please ensure that 'bins' is properly initialized.")
        if self.imageProcessor is None:
            raise ValueError("Image is not initialised")

        self.folder_path=self.imageProcessor.getImageFolder()
        self.csv_filename = os.path.join(self.folder_path, f"{self.sampleID}.csv")

        self.p.setdiameter_threshold(10)
        self.p.save_masks_to_csv(self.csv_filename)
        print(f"--> Masks saved to CSV file: {self.csv_filename}")

        self.savePsdData()
        
    def formatResults(self):
        self.totArea=self.p.get_totalArea()
        print("-----------------------------------------------")
        print("Sample ID:", self.sampleID)
        print("Total Area:", self.totArea)
        print("Scaling Factor:", self.ContainerScaler.scalingFactor)
        print("Scaling Number:", self.ContainerScaler.scalingNumber)
        self.intensity=self.imageProcessor.getIntensity()
        print("Intensity:", self.intensity)
        print("Scaling Stamp:", self.ContainerScaler.scalingStamp)
        print("Analysis Time:", self.analysisTime)
        print("Number of Particles:", self.numberofBins)
        print("diameter Threshold:", self.p.diameter_threshold)
        print("circularity Threshold:", self.p.circularity_threshold)
        print("-----------------------------------------------")

        formatter = sa.sizeAnalysisModel(self.sampleID,self.csv_filename, self.distributions_filename,self.totArea,self.ContainerScaler.scalingNumber,\
                                        self.ContainerScaler.scalingFactor,self.ContainerScaler.scalingStamp,self.intensity,self.analysisTime, \
                                        self.p.diameter_threshold,self.p.circularity_threshold)
        formatter.save_xml()
    
    def saveSegments(self):
        #save segments in json file
        self.json_filename=os.path.join(self.folder_path, f"{self.sampleID}_segments.txt")
        self.p.save_segments(self.json_filename)
        print(f"Saving segments in {self.json_filename}")
    
    def loadSegments(self,checkpoint_folder,bins):
        #load segments from json file in case no gpu available  - save them onto csv and _distribution.txt files
        try:
            self.setFolderPath()
            self.json_masks_filename = os.path.join(self.folder_path, f"{self.sampleID}_segments.txt")
            
            if not os.path.exists(self.json_masks_filename):
                raise FileNotFoundError(f"The file {self.json_masks_filename} was not found.")
            
            self.loadModel(checkpoint_folder)
            self.setBins(bins)
            self.csv_filename = os.path.join(self.folder_path, f"{self.sampleID}.csv")
            self.p.save_segments_as_csv(self.json_masks_filename, self.csv_filename)
            self.savePsdData()
        
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
    """
    def saveMasks(self):
        #save mask in json file
        self.json_masks_filename=os.path.join(self.folder_path, f"{self.sampleID}_masks.txt")
        self.p.save_masked_regions(self.json_masks_filename)
        print(f"Saving segments in {self.json_masks_filename}")

    def loadMasks(self,checkpoint_folder,bins):
        #load segments from json file in case no gpu available  
        self.setFolderPath()
        
        self.json_masks_filename=os.path.join(self.folder_path, f"{self.sampleID}_masks.txt")
        self.loadModel(checkpoint_folder)
        self.setBins(bins)
        self.p.open_segments(self.json_masks_filename)
    """
    def setFolderPath(self):
        if self.imageProcessor is not None:
            self.folder_path=self.imageProcessor.getImageFolder()
        else:
            raise ValueError("Image not initialized. Please ensure that 'imageProcessor' is properly initialized.")
        
        
        

    
    
    





