import cv2


class ROISelector:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Image at {image_path} could not be loaded.")
        # Resize for ROI selection
        self.image = cv2.resize(self.original_image, (512, 512))
        self.original_height, self.original_width = self.original_image.shape[:2]
        self.resized_height, self.resized_width = self.image.shape[:2]
        self.roi = None
        self.drawing = False
        self.rect_start = None
        self.rect_end = None

    def draw_rectangle(self, event, x, y):
        """Handles mouse events to draw the ROI."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rect_start = (x, y)
            self.rect_end = None  # Reset rect_end when a new ROI selection starts

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Reset image to original size each time
                self.image = cv2.resize(self.original_image, (512, 512))
                cv2.rectangle(self.image, self.rect_start,
                              (x, y), (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect_end = (x, y)
            cv2.rectangle(self.image, self.rect_start,
                          self.rect_end, (0, 255, 0), 2)

    def select_and_move_roi(self):
        """Display the image and allow the user to select and move the ROI."""
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", self.draw_rectangle)

        while True:
            cv2.imshow("Select ROI", self.image)
            key = cv2.waitKey(1) & 0xFF

            # If the user presses Enter (13), save the cropped image
            if key == 13:  # Enter key
                if self.rect_start and self.rect_end:
                    x1, y1 = self.rect_start
                    x2, y2 = self.rect_end

                    # Ensure valid ROI coordinates by swapping if necessary
                    x1, x2 = sorted([x1, x2])  # Ensure x1 < x2
                    y1, y2 = sorted([y1, y2])  # Ensure y1 < y2

                    # Scale the rectangle back to the original image size
                    x1_orig = int(x1 * self.original_width /
                                  self.resized_width)
                    y1_orig = int(y1 * self.original_height /
                                  self.resized_height)
                    x2_orig = int(x2 * self.original_width /
                                  self.resized_width)
                    y2_orig = int(y2 * self.original_height /
                                  self.resized_height)

                    # Ensure the coordinates are within bounds
                    x1_orig = max(0, min(x1_orig, self.original_width - 1))
                    y1_orig = max(0, min(y1_orig, self.original_height - 1))
                    x2_orig = max(0, min(x2_orig, self.original_width - 1))
                    y2_orig = max(0, min(y2_orig, self.original_height - 1))

                    # Crop the original image using the adjusted coordinates
                    cropped_image = self.original_image[y1_orig:y2_orig,
                                                        x1_orig:x2_orig]

                    # Check if cropped image is empty
                    if cropped_image.size == 0:
                        print("Cropped image is empty. Please select a valid region.")
                        continue
                    break

            # If the user presses 'q', quit without saving
            elif key == ord('q'):
                print("Exiting without saving.")
                break

        cv2.destroyAllWindows()
        return cropped_image if self.rect_start and self.rect_end else None
