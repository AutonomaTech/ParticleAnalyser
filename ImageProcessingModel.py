import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil

class ImageProcessingModel:
    def __init__(self, image_folder_path, sampleID):
        self.sampleID = sampleID
        self.image_folder_path = image_folder_path
        self.file_extensions = ['.png', '.bmp']
        self.imagePath = None
        self.image_extension = None
        self.raw_imagePath = None  # Attribute to store the path of the raw image copy

        # Loop through extensions and check for existence
        for ext in self.file_extensions:
            self.imageName = f"{self.sampleID}{ext}"
            self.imagePath = os.path.join(image_folder_path, self.imageName)
                
            if os.path.exists(self.imagePath):
                self.image_extension = ext
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
        
        final_image_name="final_"+self.sampleID + ".png"
        self.imagePath=os.path.join(self.image_folder_path,final_image_name)
        final_image.save(self.imagePath)
        print(f"Final overlaid image saved as: {self.imagePath}")

    def even_out_lighting(self):
        # Load the image
        image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)

        # Convert to LAB color space to separate intensity from color information
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split the LAB image into its channels
        l, a, b = cv2.split(lab_image)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))  # Increased grid size for less aggressive enhancement
        l_clahe = clahe.apply(l)
        
        # Merge the channels back
        lab_clahe = cv2.merge((l_clahe, a, b))
        
        # Convert back to BGR color space
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Perform a light normalization to smooth out lighting inconsistencies without over-smoothing
        enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply a slight Gaussian blur to avoid too much noise while keeping details
        final_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        self.imagePath=os.path.join(self.image_folder_path,f"even_lighting_{self.imageName}")
        # Save the final image after light normalization to the original image path
        cv2.imwrite(self.imagePath, final_image)
        print(f"Evened out lighting picture saved as : {self.imagePath}")

    def cropImage(self):
        """
        Allows the user to crop an image using a graphical ROI (Region of Interest) selection.
        
        :param image_path: Path to the image.
        :return: Cropped image.
        """
        # Read the image
        image = cv2.imread(self.imagePath)

        # Display the image and let the user select a region of interest (ROI)
        print("Select the region of interest (ROI) to crop.")
        roi = cv2.selectROI("Select ROI", image)

        # Crop the image using the selected ROI (x, y, width, height)
        cropped_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

        # Close the ROI window after selection
        cv2.destroyAllWindows()

        # Return the cropped image
        return cropped_image

    def save_cropped_image(self,cropped_image, output_path='cropped_image.jpg'):
        """
        Saves and displays the cropped image.
        
        :param cropped_image: The cropped image to save and display.
        :param output_path: The path where the cropped image will be saved.
        
        # Save the cropped image to the specified output path
        cv2.imwrite(output_path, cropped_image)------------------------

        # Display the cropped image
        cv2.imshow("Cropped Image", cropped_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
        """


