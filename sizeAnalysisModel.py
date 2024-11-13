import xml.etree.ElementTree as ET
import logger_config
import csv
from xml.dom import minidom
from datetime import datetime

logger = logger_config.get_logger(__name__)


class sizeAnalysisModel:
    def __init__(self, sampleId,sampleIdFilePath=None, psdFilePath=None, tot_area=None, scaling_num=None, scaling_fact=None, scaling_stamp=None, intensity=None,
                 analysis_time=None, bin_num=None,minmum_area=None):

        self.tot_area = tot_area
        self.segments_file_path = sampleIdFilePath
        self.psd_file_path = psdFilePath
        self.minmum_area=minmum_area
        self.scaling_num = scaling_num
        self.scaling_fact = scaling_fact
        self.scaling_stamp = scaling_stamp
        self.intensity = intensity
        self.date_time = ""
        self.analysis_time = analysis_time
        self.rows = bin_num
        self.particles = []
        self.sampleId = sampleId
        self.over_s_value = 0
        self.under_s_value = 0
        self.d_10 = 0
        self.d_50 = 0
        self.d_90 = 0
        self.mean_size = 0
        self.passing = []
        self.retaining = []
        self.xmlstring=""


    # Todo--processing file name to grab sample id
    def __retrieveSampleId(self):
        """
        Retrieving Sample Id and then compare with Sample ids the specified folders, if sample Id not existed in that file
        Keep processing,if already existed in the file, then skipped ouver
        """
        # try:
        #     if self.segments_file_path.endswith('.csv'):

        #         self.sampleId= self.segments_file_path.split('.csv')[0]

        # except:
        #         self.sampleId=""
        # finally:
        logger.info("For sampleId: {}, tot_area:{}, scaling_num:{},scaling_fact:{},scaling_stamp:{},intensity:{},analysis_time:{},bin_num:{} ",
                        self.sampleId,self.tot_area,self.scaling_num,self.scaling_fact,self.scaling_stamp,self.intensity,self.analysis_time,
                        self.rows)



    def __countNumParticles(self):
        """
        This function counts the number of data rows in a given file, ignoring the header row.

        Args:
        file_path (str): The path to the file.

        Returns:
        int: The number of data rows in the file.
        """
        try:
            with open(self.segments_file_path, 'r') as file:
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
                self.__countUnderSValue()
                self.__countOverSValue()
                self.__countMeanSize()
                self.__countD90()
                self.__countD10()
                self.__countD50()


        except :
                logger.error("Segments csv file can  not be parsed")

    def __countOverSValue(self):
        """
        This function counts  OverS (8) [%] value
        """
        overSValue = 0
        overSValuePercentage = 0
        if len(self.particles) == 0:
            logger.error("There is no particles for OverS (8) [%] value to be processed")
            return

        diameters = [particle['diameter'] for particle in self.particles]

        for diameter in diameters:
            if diameter / 1000 > 8:
                overSValue += 1

        if overSValue > 0:
            overSValuePercentage = overSValue / len(self.particles) * 100

        overSValuePercentage=  format(max(float( overSValuePercentage), 0), '.8f' )

        logger.info("OverS (8) [%]: {}", overSValuePercentage)

        self.over_s_value = overSValuePercentage

    def __countUnderSValue(self):
        """
        This function counts UnderS (0.15) [%] value

        """
        if len(self.particles) == 0:
            logger.error("There is no particles for UnderS (0.15) [%] value to be processed")
            return

        underSValue = 0
        underSValuePercentage = 0

        diameters = [particle['diameter'] for particle in self.particles]

        for diameter in diameters:
            if diameter / 1000 < 0.15:
                underSValue += 1

        if underSValue > 0:
            underSValuePercentage = underSValue / len(self.particles) * 100

        underSValuePercentage = format(max(float(underSValuePercentage), 0), '.8f')

        logger.info("UnderS (0.15) [%]: {}", underSValuePercentage)

        self.under_s_value = underSValuePercentage

    def __countMeanSize(self):
        """
        This function counts the mean size of all the particles,based on diameter first
        """
        if len(self.particles) == 0:
            logger.error("There is no particles for mean size to be processed")
            return

        totalSize = 0

        diameters = [particle['diameter'] for particle in self.particles]

        for diameter in diameters:
            totalSize += diameter

        meanSize = format(max(float( totalSize / len(self.particles) / 1000), 0), '.8f' )

        logger.info("Mean Size : {}", meanSize)

        self.mean_size = meanSize

    def __countD10(self):
        """
        This function counts D10 indicates that 10% of all particles have a diameter that is less than or equal to this value
        """
        if len(self.particles) == 0:
            logger.error("There is no particles for D0 to be processed")
            return

        count_10_per = int(len(self.particles) * 0.1)

        diameters = [particle['diameter'] for particle in self.particles]
        sorted_diameters = sorted(diameters)
        self.d_10 =format(max(float(sorted_diameters[count_10_per]/1000), 0), '.8f' )

        logger.info("D10mm : {}", self.d_10)

    def __countD50(self):
        """
        This function counts D10 indicates that 10% of all particles have a diameter that is less than or equal to this value
        """
        if len(self.particles) == 0:
            logger.error("There is no particles for D50 to be processed")
            return

        count_50_per = int(len(self.particles) * 0.5)

        diameters = [particle['diameter'] for particle in self.particles]
        sorted_diameters = sorted(diameters)

        self.d_50 =format(max(float(sorted_diameters[count_50_per]/1000), 0), '.8f' )

        logger.info("D50mm : {}", self.d_50)

    def __countD90(self):
        """
        This function counts D10 indicates that 10% of all particles have a diameter that is less than or equal to this value
        """
        if len(self.particles) == 0:
            logger.error("There is no particles for D90 to be processed")
            return

        count_90_per = int(len(self.particles) * 0.9)

        diameters = [particle['diameter'] for particle in self.particles]
        sorted_diameters = sorted(diameters)
        self.d_90 =format(max(float(sorted_diameters[count_90_per]/1000), 0), '.8f' )

        logger.info("D90mm : {}", self.d_90)

    # Todo--fileFormat confirming
    def __filterDistribution(self):
        """
            This function counts the distributions for passing and retaining
            """
        input_string = ''
        try:
        # Take string in the psd file
            with open(self.psd_file_path, 'r') as file:

                for line in file:
                    input_string = line
            # Split the input string by comma
            elements = input_string.split(',')

            # Filter the bins from the input; these are assumed to be the values before "% Passing"
            bin_array = elements[1:elements.index('Bottom') + 1]

            # Find the indices for passing and retaining percentages
            passing_start = elements.index('% Passing') + 1
            retaining_start = elements.index('% Retained') + 1

            # Extract the passing and retaining percentages
            passing_raw = elements[passing_start:passing_start + len(bin_array)]
            retaining_raw = elements[retaining_start:retaining_start + len(bin_array)]

            # Ensure the lengths of arrays match bins array
            # If the length of passing or retaining is less than bins, pad with '0.0'
            if len(passing_raw) < len(bin_array):
                passing_raw += ['0.0'] * (len(bin_array) - len(passing_raw))
            if len(retaining_raw) < len(bin_array):
                retaining_raw += ['0.0'] * (len(bin_array) - len(retaining_raw))

            # Convert lists to integer arrays, formatting floats to 8 decimal places
            # and replacing negative numbers with zero
            passing = [format(max(float(num), 0), '.8f') for num in passing_raw]
            retaining = [format(max(float(num), 0), '.8f') for num in retaining_raw]

            self.passing = passing
            self.retaining = retaining
        except:
            logger.error("Distribution file can not be parsed ")

    def __generate_iso_datetime(self):

        now = datetime.now()

        self.date_time= now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    def __build_xml(self):
        # CaCulate all the data
        # self.__retrieveSampleId()
        # if self.sampleId=="":
        #     logger.info("No sampleId is retrieved")
        #     return
        self.__countNumParticles()
        # self.__countUnderSValue()
        # self.__countOverSValue()
        # self.__countMeanSize()
        # self.__countD90()
        # self.__countD10()
        # self.__countD50()
        self.__filterDistribution()
        self.__generate_iso_datetime()

        # Create root  node of the xml file
        root = ET.Element('IAResult')

        # Adding leaf node to the root node ，including all kinds of information
        ET.SubElement(root, 'SampleID').text = str(self.sampleId)
        ET.SubElement(root, 'CustomField1')
        ET.SubElement(root, 'CustomField2')
        ET.SubElement(root, 'CustomField3')
        ET.SubElement(root, 'CustomField4')
        ET.SubElement(root, 'CustomField5')
        ET.SubElement(root, 'NumParticles').text = str(len(self.particles))
        ET.SubElement(root, 'TotArea').text = str(self.tot_area)
        ET.SubElement(root, 'ScalingNum').text = str(self.scaling_num)
        ET.SubElement(root, 'ScalingStamp').text = self.scaling_stamp
        ET.SubElement(root, 'Intensity').text = str(self.intensity)
        ET.SubElement(root, 'DateTime').text = self.date_time
        ET.SubElement(root, 'AnalysisTime').text = self.analysis_time
        ET.SubElement(root, 'NumResultTables').text=str(1)
        ET.SubElement(root, 'NumSummaryData').text=str(8)
        result_table = ET.SubElement(root, 'ResultTable')
        ET.SubElement(result_table, 'TableId').text = "1"
        ET.SubElement(result_table, 'NumColumns').text = "2"
        result_columns_passing = ET.SubElement(result_table, 'ResultColumns')
        self.add_result_columns(result_columns_passing, '1', '% Passing', 'Passing', self.passing)
        result_columns_retained = ET.SubElement(result_table, 'ResultColumns')
        self.add_result_columns(result_columns_retained, '2', '% Retained', 'Retained', self.retaining)
        # Adding summary D_10 data node
        D10_summary_data = ET.SubElement(root, 'SummaryData')
        ET.SubElement(D10_summary_data, 'Id').text = str(3)
        ET.SubElement(D10_summary_data, 'ColumnId').text = str(3)
        ET.SubElement(D10_summary_data, 'Name').text = "D10 [mm]"
        ET.SubElement(D10_summary_data, 'Value').text = str(self.d_10)
        # Adding summary D_50 data node
        D50_summary_data = ET.SubElement(root, 'SummaryData')
        ET.SubElement(D50_summary_data, 'Id').text = str(1)
        ET.SubElement(D50_summary_data, 'ColumnId').text = str(1)
        ET.SubElement(D50_summary_data, 'Name').text = "D50 [mm]"
        ET.SubElement(D50_summary_data, 'Value').text = str(self.d_50)
        # Adding summary D_90 data node
        D90_summary_data = ET.SubElement(root, 'SummaryData')
        ET.SubElement(D90_summary_data, 'Id').text = str(2)
        ET.SubElement(D90_summary_data, 'ColumnId').text = str(2)
        ET.SubElement(D90_summary_data, 'Name').text = "D90 [mm]"
        ET.SubElement(D90_summary_data, 'Value').text = str(self.d_90)
        # Adding summary overs  data node
        Overs_summary_data = ET.SubElement(root, 'SummaryData')
        ET.SubElement(Overs_summary_data, 'Id').text = str(6)
        ET.SubElement(Overs_summary_data, 'ColumnId').text = str(6)
        ET.SubElement(Overs_summary_data, 'Name').text = "OverS (8) [%]"
        ET.SubElement(Overs_summary_data, 'Value').text = str(self.over_s_value)
        # Adding summary under s  data node
        Unders_summary_data = ET.SubElement(root, 'SummaryData')
        ET.SubElement(Unders_summary_data, 'Id').text = str(7)
        ET.SubElement(Unders_summary_data, 'ColumnId').text = str(7)
        ET.SubElement(Unders_summary_data, 'Name').text = "UnderS (0.15) [%]"
        ET.SubElement(Unders_summary_data, 'Value').text = str(self.under_s_value)

        # Adding summary total  data node
        Total_Part = ET.SubElement(root, 'SummaryData')
        ET.SubElement(Total_Part, 'Id').text = str(7)
        ET.SubElement(Total_Part, 'ColumnId').text = str(7)
        ET.SubElement(Total_Part, 'Name').text = "Total Part."
        ET.SubElement(Total_Part, 'Value').text = str(len(self.particles))

        # Adding Mean Size
        Mean_Size = ET.SubElement(root, 'SummaryData')
        ET.SubElement(Mean_Size, 'Id').text = str(7)
        ET.SubElement(Mean_Size, 'ColumnId').text = str(7)
        ET.SubElement(Mean_Size, 'Name').text = "Mean Size"
        ET.SubElement(Mean_Size, 'Value').text = str(self.mean_size)
        self.xmlstring= ET.tostring(root, encoding='unicode', method='xml')






    def add_result_columns(self, parent, column_id, class_desc, dist, distribution):
        ET.SubElement(parent, 'ColumnId').text = column_id
        ET.SubElement(parent, 'ClassDesc').text = class_desc
        ET.SubElement(parent, 'DistDesc').text = 'Area %'
        ET.SubElement(parent, 'Dist').text = dist
        ET.SubElement(parent, 'Unit').text = 'mm'
        ET.SubElement(parent, 'MeshType').text = 'Mesh_MM'
        ET.SubElement(parent, 'NumRows').text = str(self.rows)

        sieve_desc = ET.SubElement(parent, 'SieveDesc')
        sieve_size = ET.SubElement(parent, 'SieveSize')
        for desc, size in zip(['8', '1', '0.106', '0.038', 'Bottom'],
                              ['8.00000000', '1.00000000', '0.10600000', '0.03800001', '0.00000000']):
            ET.SubElement(sieve_desc, 'Value').text = desc
            ET.SubElement(sieve_size, 'Value').text = size

        distribution_node = ET.SubElement(parent, 'Distribution')
        for value in distribution:
            ET.SubElement(distribution_node, 'Value').text = str(value)

    def save_xml(self):
        self.__build_xml()

        if self.xmlstring=="":
            logger.info("No xml file is generated")
            return

        reparsed = minidom.parseString(self.xmlstring)
        pretty_string = reparsed.toprettyxml(indent="  ")
        pretty_string_without_declaration = '\n'.join(pretty_string.split('\n')[1:])
        filename = f"{self.sampleId}.xml"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(pretty_string_without_declaration)
