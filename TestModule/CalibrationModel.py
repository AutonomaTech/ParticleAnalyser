
import logger_config
logger = logger_config.get_logger(__name__)
import os
import math
import bisect
import configparser
import csv
import matplotlib.pyplot as plt
import numpy as np

class CalibrationModel:
    def __init__(self, totArea=None,csv_filename=None,folder_path=None,sampleId=None,bins=None):
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
        self.bins=bins
        self.ini_file_path = "calibration_bin.ini"
        self.config = configparser.ConfigParser()
        self.load_config()

    def load_config(self):
        if os.path.exists(self.ini_file_path):
            self.config.read(self.ini_file_path)
            existing_bins = [int(section) for section in self.config.sections() if section.isdigit()]
            self.new_bin_number = max(existing_bins, default=0) + 1
        else:
            self.new_bin_number = 1

    def calibrate_bin_with_size(self, target_distribution=None):

        # Confirm new section number

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


        for percentage in target_distribution[:-1]:  # The last bin does not require additional calculation
            cumulative_percentage += percentage
            index = int(
                cumulative_percentage / 100 * total_particles)  # Locate the index corresponding to the target percentage
            bin_value = diameters[min(index, total_particles - 1)]  # Ensure the index does not go out of bounds
            self.calibrated_bins_with_size.append(math.ceil(bin_value))

        self.config[str(self.new_bin_number)] = {}
        self.config[str(self.new_bin_number)]['bySize'] = f"[{', '.join(map(str, self.calibrated_bins_with_size))}]"
        self.save_config()

        return self.calibrated_bins_with_size

    def calibrate_bin_with_area(self, target_distribution=None):

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


        cumulative_percentage = 0.0

        for i, percentage in enumerate(target_distribution[:-1]):  # Exclude the last percentage
            cumulative_percentage += percentage
            diameter = self.find_cumulative_area_diameter(self.particles, total_area, cumulative_percentage)
            self.calibrated_bins_with_area.append(diameter)

        self.config[str(self.new_bin_number)] = {}
        self.config[str(self.new_bin_number)]['byArea'] = f"[{', '.join(map(str, self.calibrated_bins_with_area))}]"
        self.save_config()

        return self.calibrated_bins_with_area

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

    def find_cumulative_area_diameter_new(self, particles, identified_area, unsegmented_area, target_percentage):
        """
        考虑 unsegmented_area 占比较大的情况下的新算法
        """
        print("\n" + "=" * 50)
        print("开始新的bin计算")

        total_area = identified_area + unsegmented_area
        print(f"\n基础面积信息:")
        print(f"Identified area: {identified_area}")
        print(f"Unsegmented area: {unsegmented_area}")
        print(f"Total area: {total_area}")

        # 计算已识别区域占总面积的比例
        identified_ratio = identified_area / total_area
        print(f"\n面积比例:")
        print(f"Identified area ratio: {identified_ratio * 100:.2f}%")
        print(f"Unsegmented area ratio: {((unsegmented_area / total_area) * 100):.2f}%")

        # 调整目标百分比
        adjusted_target_percentage = (target_percentage * identified_area) / total_area
        print(f"\n目标百分比调整:")
        print(f"Original target percentage: {target_percentage:.2f}%")
        print(f"Adjusted target percentage: {adjusted_target_percentage:.2f}%")

        particles_sorted = sorted(particles, key=lambda x: x['diameter'], reverse=True)
        print(f"\n粒子直径范围:")
        print(f"Largest diameter: {particles_sorted[0]['diameter']:.2f}")
        print(f"Smallest diameter: {particles_sorted[-1]['diameter']:.2f}")

        cumulative_sum = 0.0
        target_area = identified_area * (adjusted_target_percentage / 100)
        print(f"\n目标面积:")
        print(f"Target area to reach: {target_area:.2f}")

        # 跟踪计算过程
        for particle in particles_sorted:
            cumulative_sum += particle['area']
            current_percentage = (cumulative_sum / identified_area) * 100
            print(f"\n当前累积:")
            print(f"Current diameter: {particle['diameter']:.2f}")
            print(f"Current cumulative area: {cumulative_sum:.2f}")
            print(f"Current percentage of identified area: {current_percentage:.2f}%")

            if cumulative_sum >= target_area:
                print(f"\n找到目标直径: {particle['diameter']:.2f}")
                return round(particle['diameter'])

        print("\n未找到满足条件的直径，使用最小直径作为返回值")
        return round(particles_sorted[-1]['diameter']) if particles_sorted else None

    def calculate_unsegmented_area(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        # Extract containerWidth directly in micrometers (um)
        container_width_um = float(config['analysis']['containerWidth'])

        self.container_area_um2 = container_width_um ** 2  # Assuming the container is a square


        if self.totArea is not None:  # Check if totArea has been set
            if self.totArea < self.container_area_um2:
                self.unSegmentedArea = self.container_area_um2 - self.totArea


        print(f"Total area is {self.totArea}.")
        print(f"Container Width (um): {container_width_um}")
        print(f"Container Area (um²): {self.container_area_um2}")
        print(f"Unsegmented Area (um²): {self.unSegmentedArea}")
        return self.unSegmentedArea


    def calibrate_bin_with_area_updated(self, target_distribution=None):

        """
          Calibrate bins based on area distribution, including unsegmented areas.
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
        if not self.particles:
            return

        # Calculate total area including unsegmented area
        identified_area = sum(particle['area'] for particle in self.particles)
        total_area = identified_area + self.unSegmentedArea

        cumulative_percentage = 0.0
        self.calibrated_bins_with_area = []

        for percentage in target_distribution[:-1]:  # Exclude the last percentage
            cumulative_percentage += percentage
            diameter = self.find_cumulative_area_diameter_new(
                self.particles,
                identified_area,
                self.unSegmentedArea,
                cumulative_percentage
            )
            if diameter is not None:
                self.calibrated_bins_with_area.append(diameter)

        # Save configuration
        print(f"calibrated_bins_with_area:{self.calibrated_bins_with_area}")
        self.config[str(self.new_bin_number)] = {}
        self.config[str(self.new_bin_number)]['byArea'] = f"[{', '.join(map(str, self.calibrated_bins_with_area))}]"
        self.save_config()

        return self.calibrated_bins_with_area

    def calibrated_bins_with_unSegementedArea(self):


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
        minBin = self.bins[0]
        new_standardBin = []

        diameters = [particle['diameter'] for particle in self.particles]
        sorted_diameters = sorted(diameters)
        minimum_diameter = sorted_diameters[0]
        if minimum_diameter < minBin:
            minBin = round(minimum_diameter)
            new_standardBin = [minBin] + self.bins
        if minimum_diameter > minBin:
            bisect.insort(self.bins, round(minimum_diameter))
            new_standardBin = self.bins
        self.calibrated_bins_with_unsegementArea = new_standardBin
        return self.calibrated_bins_with_unsegementArea

    def refactorPSD(self, unsegmentedArea, calibrated_bins, container_area):
            """
              Refactor PSD data with unsegmented area
            """
            # self.calculate_unsegmented_area()
            # self.calculate_bins_with_unsegementedArea()
            if len(calibrated_bins) == 0 or unsegmentedArea == 0:
                print("newStandardBins or unsegmented area not existed")
                return
            newCount = unsegmentedArea / container_area * 100
            distributions_filename = os.path.join(self.folder_path, f'{self.sampleID}_byArea_distribution.txt')
            input_string = ''

            # Take string in the psd file
            with open(distributions_filename, 'r') as file:

                for line in file:
                    input_string = line
            # Split the input string by commxa
            elements = input_string.split(',')
            bin_array = elements[1:elements.index('Bottom') + 1]

            passing_start = elements.index('% Passing') + 1
            passing_end = elements.index('% Retained')
            retaining_start = elements.index('% Retained') + 1

            # Extract the passing and retaining percentages--only 4 values will be produced
            passing_raw = elements[passing_start:passing_end]

            retaining_raw = elements[retaining_start:]

            # If we have new bins with unsegemented area

            if len(calibrated_bins) > 0:
                newBinarray = sorted(calibrated_bins, reverse=True)
                newBinarray.append(bin_array[-1])

                new_retaining = self.update_retaining(bin_array, newBinarray, retaining_raw, newCount, container_area)
                new_passing = self.update_passing(new_retaining)
                refactor_csvPath = os.path.join(self.folder_path, f'{self.sampleID}_refactored_distribution.txt')
                with open(refactor_csvPath, 'w', newline='') as csvfile:
                    data = [self.sampleID] + newBinarray + ['% Passing'] + \
                           new_passing + ['% Retained'] + new_retaining
                    writer = csv.writer(csvfile)
                    writer.writerow(data)
                return newBinarray, new_retaining, new_passing


            else:
                return None

    def update_retaining(self, old_bins, new_bins, old_retaining, count, container_area):
            print("Call update retaining function")
            # Assuming the second-to-last element in old_bins, excluding 'Bottom'
            key_bin = old_bins[-2]
            print("key_bin:", key_bin)

            # Find the position of the key_bin in the new_bins
            key_bin_position = new_bins.index(int(key_bin))
            # Recalculate old_retaining based on the new total area
            # Convert percentage to area using the old total area (totArea)
            areas = [float(value) * self.totArea / 100 for value in old_retaining]
            # Recalculate percentages using the new total area (containerArea)
            new_old_retaining = [area / container_area * 100 for area in areas]
            # Create new retaining based on the position of key_bin in new_bins
            if key_bin_position == len(new_bins) - 3:
                # If key_bin is third-to-last, append count directly before 'Bottom'
                new_retaining = new_old_retaining[:] + [count]
            elif key_bin_position == len(new_bins) - 2:
                # If key_bin is second-to-last, insert count just before the last element
                new_retaining = new_old_retaining[:-1] + [count] + [new_old_retaining[-1]]

            # Print results to verify
            print("Old Bin Array:", old_bins)
            print("New Bin Array:", new_bins)
            print("Old Retaining:", old_retaining)
            print("New Retaining:", new_retaining)
            return new_retaining

    def update_passing(self, new_retaining):
            cumulative_area = []
            for i, count in enumerate(new_retaining):
                if i == 0:
                    cumulative_area.append(100 - float(count))
                else:
                    cumulative_area.append(cumulative_area[i - 1] - float(count))
            print("New Passing:", cumulative_area)
            return cumulative_area

    def refactor_psd(self):
            # Obtain particle size distribution data
            result = self.refactorPSD(self.unSegmentedArea, self.calibrated_bins_with_unsegementArea,self.container_area_um2)

            if result is None:
                return

            bin_edges, counts, cumulative_area = result
            # Reverse cumulative_area and skip the first data point
            cumulative_area = cumulative_area[::-1][1:]  # Start from the second element

            # Skip the first data point for counts as well
            counts = counts[::-1][1:]
            bin_edges = bin_edges[:-1][::-1]  # Start from the second element

            # Create the main histogram plot
            f, ax = plt.subplots(figsize=(12, 8))
            # Add more padding to prevent cutoff
            plt.subplots_adjust(left=0.15, right=0.85, bottom=0.2, top=0.9)

            # Distribute bin_edges evenly
            equal_spacing = np.linspace(0, 1, len(counts))
            bin_width = equal_spacing[1] - equal_spacing[0]  # Calculate the width of each bin

            # Draw the bars with a width of 80% of the equal spacing to ensure gaps
            ax.bar(equal_spacing, counts, width=bin_width * 0.8, align='center', edgecolor='black', color='skyblue',label='% Retained (Area %)')

            # Set the ticks and labels for the x-axis based on bin boundaries
            ax.set_xticks(equal_spacing)
            ax.set_xticklabels([f'{edge / 1000}' for edge in bin_edges])  # Convert edge to mm

            # Create a secondary y-axis for the cumulative percentage
            ax1 = ax.twinx()
            ax1.plot(equal_spacing, cumulative_area, 'o-', color='red',
                     linewidth=2,label='Cumulative % passing (Area %)')  # Ensure points are connected by lines

            # Format the secondary y-axis as percentage
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

            # Set labels for both axes
            ax.set_xlabel('Particle size (mm)', labelpad=20)
            ax.set_ylabel('% Retained (Area %)',labelpad=20)
            ax1.set_ylabel('Cumulative % passing (Area %)',labelpad=10)
            # Get handles and labels from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax1.get_legend_handles_labels()

            # Create a unified legend below the plot
            ax1.legend(lines1 + lines2, labels1 + labels2,
                       loc='center', bbox_to_anchor=(0.5, -0.25),
                       ncol=1, frameon=True, fancybox=True, shadow=True)
            fileName = os.path.join(self.folder_path, f'{self.sampleID}_refactor_distribution.png')
            # Adjust the layout to prevent legend cutoff
            plt.tight_layout()

            # Save the plot as an image file with adjusted figure size to accommodate legend
            plt.gcf().set_size_inches(8, 7)  # Adjust figure size if needed
            plt.title("Particle size distribution", pad=20)
            plt.savefig(fileName, bbox_inches='tight', dpi=300, pad_inches=0.5)  # Save the plot to the path constructed
            plt.close()

    def save_config(self):
        with open(self.ini_file_path, 'w') as configfile:
            self.config.write(configfile)
