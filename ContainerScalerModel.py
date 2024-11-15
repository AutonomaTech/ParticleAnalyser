from datetime import datetime
class ContainerScalerModel:
    def __init__(self,containerWidth):
        self.containerWidth = containerWidth
        self.scalingFactor=None
        self.scalingNumber=None
        self.scalingStamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    
    
    def updateScalingFactor(self,scalingNumber, containerWidth=None):
        if containerWidth is not None:
            self.containerWidth=containerWidth
        self.scalingNumber=scalingNumber
        self.scalingFactor=self.containerWidth / self.scalingNumber

    def setScalingNumber(self,scalingNumber):
      self.scalingNumber=scalingNumber

