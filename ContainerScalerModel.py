from datetime import datetime
class ContainerScalerModel:
    def __init__(self, containerWidth,W_image_pixels):


        self.scalingFactor=containerWidth / W_image_pixels
        self.scalingNumber=W_image_pixels
        self.scalingStamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    
    
    def setScalingFactor(self, scalingFactor):
        self.scalingFactor=scalingFactor

    def setScalingNumber(self,scalingNumber):
      self.scalingNumber=scalingNumber