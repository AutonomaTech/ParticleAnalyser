
import logger_config
logger = logger_config.get_logger(__name__)
import os
import math
import bisect
import configparser


class CalibrationModel:
    def __init__(self, totArea=None,csv_filename=None,folder_path=None,sampleId=None):
        self.totArea = totArea
        self.csv_filename = ""
        self.calibrated_bins_with_unsegementArea = []
        self.unSegmentedArea = 0
        self.container_area_um2 = 0
        self.csv_filename=csv_filename
        self.folder_path=folder_path
        self.sampleID=sampleId
        self.calibrated_bins_with_area=[]
        self.calibrated_bins_with_size = []
        self.particles=[]

    def getTheCalculatedBinsBySize(self, target_distribution=None):
        """
        Calculate the adjusted bins aligns with lab result
        Args:

            segmentsFilePath: The segement.txt file path
            target_distribution: Laboratory Screen Percentages (target distribution).
        Returns:
            bins: Adjusted bins that will fit the Laboratory Screen Percentages
        """

        if len(self.particles) == 0:

            if not os.path.exists(self.csv_filename):
                return
            with open(self.csv_filename, 'r') as file:
                next(file)
                for line in file:
                    if line.strip():  # remove white space
                        area, perimeter, diameter, circularity = map(float, line.strip().split(','))
                        item = {
                            "area": area,
                            "perimeter": perimeter,
                            "diameter": diameter,
                            "circularity": circularity
                        }
                        self.particles.append(item)
        if len(self.particles) == 0:
            return
        cumulative_percentage = 0.0
        diameters = [particle['diameter'] for particle in self.particles]
        diameters = sorted(diameters, reverse=True)
        # Sort diameters in ascending order
        total_particles = len(diameters)
        bins = []

        for percentage in target_distribution[:-1]:  # The last bin does not require additional calculation
            cumulative_percentage += percentage
            index = int(
                cumulative_percentage / 100 * total_particles)  # Locate the index corresponding to the target percentage
            bin_value = diameters[min(index, total_particles - 1)]  # Ensure the index does not go out of bounds
            bins.append(math.ceil(bin_value))
        # Step 4: Add the maximum value of the last bin

        # Save the bins to a text file
        bins_file_path = os.path.join(self.folder_path, f"{self.sampleID}_bins_calibrated_size.txt")
        with open(bins_file_path, 'w') as file:
            file.write(', '.join(map(str, bins)))

    def calculate_cumulative_bins_byArea(self, target_distribution=None):
        """
        Calculate the diameters for cumulative percentage thresholds.

        Args:
            particles (list of dicts): List containing particle data with 'diameter' and 'area'.
            total_area (float): The total area of all particles.
            target_percentages (list): List of target percentages for cumulative calculations.

        Returns:
            dict: Dictionary where keys are cumulative percentages and values are the corresponding diameters.
        """
        if len(self.particles) == 0:

            if not os.path.exists(self.csv_filename):
                return

            with open(self.csv_filename, 'r') as file:
                next(file)
                for line in file:
                    if line.strip():  # remove white space
                        area, perimeter, diameter, circularity = map(float, line.strip().split(','))
                        item = {
                            "area": area,
                            "perimeter": perimeter,
                            "diameter": diameter,
                            "circularity": circularity
                        }
                        self.particles.append(item)
        total_area = 0
        if len(self.particles) == 0:
            return
        areas = [particle['area'] for particle in self.particles]
        for area in areas:
            total_area += area

        bins = []
        cumulative_percentage = 0.0

        for i, percentage in enumerate(target_distribution[:-1]):  # Exclude the last percentage
            cumulative_percentage += percentage
            diameter = self.find_cumulative_area_diameter(self.particles, total_area, cumulative_percentage)
            bins.append(diameter)
        bins_file_path = os.path.join(self.folder_path, f"{self.sampleID}_bins_calibrated_area.txt")
        with open(bins_file_path, 'w') as file:
            file.write(', '.join(map(str, bins)))

    def find_cumulative_area_diameter(self, particles, total_area, target_percentage):
        """
        Helper function to find the diameter at which a specific cumulative area percentage is reached.
        """
        target_area = total_area * (target_percentage / 100)
        particles_sorted = sorted(particles, key=lambda x: x['diameter'], reverse=True)
        cumulative_sum = 0.0

        for particle in particles_sorted:
            cumulative_sum += particle['area']
            if cumulative_sum >= target_area:
                return round(particle['diameter'])

        return None

    def calculate_unsegmented_area(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Extract containerWidth directly in micrometers (um)
        container_width_um = float(config['analysis']['containerWidth'])

        self.container_area_um2 = container_width_um ** 2  # Assuming the container is a square


        if self.totArea is not None:  # Check if totArea has been set
            if self.totArea < self.container_area_um2:
                self.unSegmentedArea = self.container_area_um2 - self.totArea

        else:
            print("Total area (self.totArea) is not set.")
        print(f"Container Width (um): {container_width_um}")
        print(f"Container Area (um²): {self.container_area_um2}")
        print(f"Unsegmented Area (um²): {self.unSegmentedArea}")
        return self.unSegmentedArea



    def calibrated_bins_with_unSegementedArea(self):
        def parse_bins(industry_bins_string):
            # Remove non-numeric characters and split by commas
            import re
            # Assuming industry_bins_string looks something like "[38, 106, 1000, 8000]"
            # Removing brackets and spaces before splitting
            cleaned_string = re.sub(r'[\[\] ]', '', industry_bins_string)
            bins_list = cleaned_string.split(',')

            # Convert each element to an integer
            try:
                industry_bins = [int(x) for x in bins_list]
                print("Parsed industry bins:", industry_bins)
                return industry_bins
            except ValueError as e:
                print("Error converting to integer:", e)
                return []

        if len(self.particles) == 0:

            if not os.path.exists(self.csv_filename):
                return

            with open(self.csv_filename, 'r') as file:
                next(file)
                for line in file:
                    if line.strip():  # remove white space
                        area, perimeter, diameter, circularity = map(float, line.strip().split(','))
                        item = {
                            "area": area,
                            "perimeter": perimeter,
                            "diameter": diameter,
                            "circularity": circularity
                        }
                        self.particles.append(item)

        config = configparser.ConfigParser()
        config.read('config.ini')

        # Extract containerWidth directly in micrometers (um)

        # Accessing the industryBin values, assuming they are stored as a comma-separated string
        industry_bins_string = config['analysis']['industryBin']
        standardBin = parse_bins(industry_bins_string)
        if not standardBin:
            return

        if len(self.particles) == 0:
            return
        minBin = standardBin[0]
        new_standardBin = []

        diameters = [particle['diameter'] for particle in self.particles]
        sorted_diameters = sorted(diameters)
        minimum_diameter = sorted_diameters[0]
        if minimum_diameter < minBin:
            minBin = round(minimum_diameter)
            new_standardBin = [minBin] + standardBin
        if minimum_diameter > minBin:
            bisect.insort(standardBin, round(minimum_diameter))
            new_standardBin = standardBin
        self.calibrated_bins_with_unsegementArea = new_standardBin
        return self.calibrated_bins_with_unsegementArea


