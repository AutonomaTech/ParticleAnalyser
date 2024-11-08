import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class ImageProcessingModel:
  def __init__(self, image_folder_path,sampleID):
      self.sampleID = sampleID
      self.image_folder_path=image_folder_path
      file_extensions = ['.png', '.bmp']
      self.imagePath = None

      # Loop through extensions and check for existence
      for ext in file_extensions:
          self.imageName = f"{self.sampleID}{ext}"
          self.imagePath = os.path.join(image_folder_path, self.imageName)
            
          if os.path.exists(self.imagePath):
              print(f"Image found: {self.imagePath}")
              break
      else:
          # If no file with the listed extensions is found, raise an error
          raise FileNotFoundError(f"No file with extensions {file_extensions} found for {self.sampleID} in folder {image_folder_path}")
  def getImagePath(self):
      return self.imagePath

  def getImageFolder(self):
      return self.image_folder_path

  #show the image that has being analysed
  def showImage(self):
      if os.path.exists(self.imagePath):
          image = Image.open(self.imagePath)
          image = np.array(image.convert("RGB"))
          plt.imshow(image)  # Display the image
          plt.axis('off')  # Optional: Turn off the axis for a cleaner view
          plt.show()  # Show the image
      else:
          print(f"Error: Image {self.imageName} not found at {self.imagePath}")

 
  #def cropImage(self):

  #overlay images to improve border contrast of rocks
  def overlayImage(self):
      if not os.path.exists(self.imagePath):
          print(f"Error: Image {self.imageName} not found at {self.imagePath}")
          return
        
      base_image = Image.open(self.imagePath).convert("RGBA")
      final_image = base_image.copy()

      for _ in range(10):
          final_image = Image.alpha_composite(final_image, base_image)

      base_image_path = os.path.join(self.image_folder_path, f"base_image_{self.sampleID}.png")
      final_image_path = self.imagePath  
      base_image.save(base_image_path)
      print(f"Base image saved as: {base_image_path}")

      final_image.save(final_image_path)
      print(f"Final overlaid image saved as: {final_image_path}")


