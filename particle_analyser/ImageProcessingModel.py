import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class ImageProcessingModel:
  def __init__(self, image_folder_path,sampleID):
      self.sampleID = sampleID
      self.image_folder_path=image_folder_path
      self.file_extensions = ['.png', '.bmp']
      self.imagePath = None

      # Loop through extensions and check for existence
      for ext in self.file_extensions:
          self.imageName = f"{self.sampleID}{ext}"
          self.imagePath = os.path.join(image_folder_path, self.imageName)
            
          if os.path.exists(self.imagePath):
              print(f"Image found: {self.imagePath}")
              break
      else:
          # If no file with the listed extensions is found, raise an error
          raise FileNotFoundError(f"No file with extensions {self.file_extensions} found for {self.sampleID} in folder {image_folder_path}")
  
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

  #get width of image for mm/pixels ratio
  def getWidth(self):
    max_width = 0
    for filename in os.listdir(self.image_folder_path):

        if filename.startswith(self.sampleID) and any(filename.endswith(ext) for ext in self.file_extensions):
            image_path = os.path.join(self.image_folder_path, filename)
            print(image_path)

            with Image.open(image_path) as img:
                width = img.width
                if width > max_width:
                    max_width = width
    
    return max_width

  #def cropImage(self):
  def getIntensity(self):
      image = Image.open(self.imagePath)
      image_np = np.array(image)
      image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility

      # Convert the image to grayscale
      gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

      # Calculate the average intensity
      return np.mean(gray_image)
      
  #overlay images to improve border contrast of rocks
  def overlayImage(self):
    if not os.path.exists(self.imagePath):
        print(f"Error: Image {self.imageName} not found at {self.imagePath}")
        return
        
    image_size_mb = os.path.getsize(self.imagePath) / (1024 * 1024)  # Size in MB

    base_image = Image.open(self.imagePath).convert("RGBA")

    # If the image size is over 8 MB, resize it to 50% of the original dimensions
    if image_size_mb > 8:
        width, height = base_image.size
        base_image = base_image.resize((width // 2, height // 2), Image.LANCZOS)
        print(f"Image size was over 8MB, resized to {width // 2}x{height // 2}.")
    final_image = base_image.copy()

    # Overlay the image on itself 10 times
    for _ in range(10):
        final_image = Image.alpha_composite(final_image, base_image)

    # Save the base and final overlaid images
    if not self.imagePath.lower().endswith('.png'):
      base_image_path = os.path.join(self.image_folder_path, f"base_image_{self.sampleID}.png")
      base_image.save(base_image_path)
      print(f"Base image saved as: {base_image_path}")
      
    final_image_path = os.path.splitext(self.imagePath)[0] + ".png"
    final_image.save(final_image_path)
    print(f"Final overlaid image saved as: {final_image_path}")

