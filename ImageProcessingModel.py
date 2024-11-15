import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import ROISelector as ROI

class ImageProcessingModel:
    def __init__(self, image_folder_path, sampleID):
        """
        Initializes the ImageProcessingModel with the provided folder path and sample ID.
        It searches for an image file with the given sample ID and supported extensions (.png, .bmp).

        Inputs:
            image_folder_path (str): Path to the folder containing images.
            sampleID (str): The sample ID to identify the image.

        Outputs:None
        """
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
        """
        Returns the full path of the image file.

        Inputs:None

        Outputs:str: The full path to the image file.
        """
        return self.imagePath

    def getImageFolder(self):
        """
        Returns the path of the folder containing the image.

        Inputs:None

        Outputs:str: The folder path where the image is located.
        """
        return self.image_folder_path

    def showImage(self):
        """
        Displays the image using matplotlib. Converts the image to RGB and shows it.

        Inputs:None

        Outputs: Show the image
            
        """
        if os.path.exists(self.imagePath):
            image = Image.open(self.imagePath)
            image = np.array(image.convert("RGB"))
            plt.imshow(image)  # Display the image
            plt.axis('off')  # Optional: Turn off the axis for a cleaner view
            plt.show()  # Show the image
        else:
            print(f"Error: Image {self.imageName} not found at {self.imagePath}")

    def getWidth(self):
        """
        Returns the maximum width of an image in the folder with the same sample ID.
        Useful for determining the mm/pixel ratio.

        Inputs:None

        Outputs:
            int: The maximum width of the images found.
        """
        max_width = 0
        for filename in os.listdir(self.image_folder_path):
            if filename.startswith(self.sampleID) and any(filename.endswith(ext) for ext in self.file_extensions):
                image_path = os.path.join(self.image_folder_path, filename)

                with Image.open(image_path) as img:
                    width = img.width
                    if width > max_width:
                        max_width = width
        
        return max_width

    def getIntensity(self):
        """
        Calculates the average intensity (grayscale) of the image.
        
        Inputs:None

        Outputs: float: The average intensity value of the image.
        """
        image = Image.open(self.imagePath)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility

        gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        return np.mean(gray_image)
        
    def overlayImage(self):
        """
        Overlays the image on itself 10 times to improve the contrast of the image,
        especially the borders of rocks, and resizes it if the size exceeds 8MB.

        Inputs:None

        Outputs:None
        """
        if not os.path.exists(self.imagePath):
            print(f"Error: Image {self.imageName} not found at {self.imagePath}")
            return

        image_size_mb = os.path.getsize(self.imagePath) / (1024 * 1024)  # Size in MB

        base_image = Image.open(self.imagePath).convert("RGBA")

        # Resize image if size is greater than 8MB
        if image_size_mb > 8:
            scale_factor = (8 / image_size_mb) ** 0.5  # Square root to maintain aspect ratio

            width, height = base_image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            base_image = base_image.resize((new_width, new_height), Image.LANCZOS)
            print(f"Image size was over 8MB, resized to {new_width}x{new_height}.")
            
            image_size_mb = base_image.tell() / (1024 * 1024)

            # If still too large, reduce it further
            while image_size_mb > 8:
                width, height = base_image.size
                base_image = base_image.resize((width // 2, height // 2), Image.LANCZOS)
                image_size_mb = base_image.tell() / (1024 * 1024)
                print(f"Still too large, further resized to {width // 2}x{height // 2}. Current size: {image_size_mb:.2f}MB")

        final_image = base_image.copy()

        # Overlay the image on itself 10 times
        for _ in range(10):
            final_image = Image.alpha_composite(final_image, base_image)

        # Save the base and final overlaid images
        if not self.imagePath.lower().endswith('.png'):
            base_image_path = os.path.join(self.image_folder_path, f"base_image_{self.sampleID}.png")
            base_image.save(base_image_path)
            print(f"Base image saved as: {base_image_path}")

        # Save the final overlaid image with a new name
        final_image_name = "final_" + self.sampleID + ".png"
        self.imagePath = os.path.join(self.image_folder_path, final_image_name)
        final_image.save(self.imagePath)
        print(f"Final overlaid image saved as: {self.imagePath}")

    def even_out_lighting(self):
        """
        Even out the lighting in the image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        to improve the contrast and smooth out lighting inconsistencies.

        Inputs:None

        Outputs:None
        """
        # Load the image
        image = cv2.imread(self.imagePath, cv2.IMREAD_COLOR)
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)# Convert to LAB color space to separate intensity from color information
        
        l, a, b = cv2.split(lab_image)# Split the LAB image into its channels
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))  # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Perform a light normalization to smooth out lighting inconsistencies without over-smoothing
        enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply a slight Gaussian blur to avoid too much noise while keeping details
        final_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        self.imagePath=os.path.join(self.image_folder_path,f"even_lighting_{self.imageName}")
        cv2.imwrite(self.imagePath, final_image)
        print(f"Evened out lighting picture saved as : {self.imagePath}")

    def cropImage(self):
        """
        Allows the user to manually select a region of interest (ROI) and crop the image to that region.
        
        Inputs:None

        Outputs:None
        """
        roi_selector = ROI.ROISelector(self.imagePath)
        cropped_image = roi_selector.select_and_move_roi()
        self.imagePath = os.path.join(self.image_folder_path, f"cropped_image_{self.imageName}")
        # Save the cropped image to the image folder
        cv2.imwrite(self.imagePath, cropped_image)
        print(f"Cropped image picture saved as : {self.imagePath}")
