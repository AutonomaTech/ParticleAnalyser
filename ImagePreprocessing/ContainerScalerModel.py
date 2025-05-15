from datetime import datetime


class ContainerScalerModel:
    """
    A class for managing and calculating scaling factors for container dimensions.

    This class is designed to maintain and update the scaling factor for a container, 
    allowing for dynamic adjustments based on a given scaling number or updated container width.
    It also tracks the timestamp of when the scaling factor is initialized or updated.

    Attributes:
    - containerWidth (float): The width of the container.
    - scalingFactor (float): The scaling factor calculated based on the container width and scaling number.
    - scalingNumber (int or float): The number used to scale the container width.
    - scalingStamp (str): A timestamp marking the creation or latest update of the scaling parameters.

    Methods:
    - __init__(containerWidth): Initializes the object with a specified container width and records the creation timestamp.
    - updateScalingFactor(scalingNumber, containerWidth=None): Updates the scaling factor using a new scaling number 
      and optionally updates the container width.
    - setScalingNumber(scalingNumber): Sets a new scaling number for the container without recalculating the scaling factor.
    """

    def __init__(self, containerWidth):
        self.containerWidth = containerWidth
        self.scalingFactor = None
        self.scalingNumber = None
        self.scalingStamp = datetime.now().strftime(
            "%Y-%m-%dT%H:%M:%S.%f")[:-3]

    def updateScalingFactor(self, imageWidth=None, scalingNumber=None, containerWidth=None):

        if containerWidth is not None:
            self.containerWidth = containerWidth
        if containerWidth == 0:
            raise ValueError("The value of containerWidth cannot be zero.")
        if scalingNumber is not None and scalingNumber > 0:
            self.scalingNumber = scalingNumber
        else:
            self.scalingNumber=imageWidth
        self.scalingFactor = self.containerWidth / self.scalingNumber

    def setScalingNumber(self, scalingNumber):
        self.scalingNumber = scalingNumber

    def setScalingFactor(self, scalingFactor):
        self.scalingFactor = scalingFactor
