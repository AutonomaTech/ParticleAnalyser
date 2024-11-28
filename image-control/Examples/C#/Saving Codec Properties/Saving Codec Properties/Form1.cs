using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Saving_Codec_Properties
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        // Global AVICompressor object
		//<<globalcodec
        private TIS.Imaging.AviCompressor _selectedCodec;
		//>>
        /// <summary>
        /// Form_Load
        ///
        /// Gets all available codecs from ICImagingControl and
        /// put their names in the cboVideoCodec combo box.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<formload
        private void Form1_Load(object sender, EventArgs e)
        {
            // Insert all installed codecs into the cboVideoCodec combobox.
            foreach (TIS.Imaging.AviCompressor codec in TIS.Imaging.AviCompressor.AviCompressors)
            {
                cboVideoCodec.Items.Add(codec);
            }
            // Show the first codec in the combobox.
            cboVideoCodec.SelectedIndex = 0;

            _selectedCodec = (TIS.Imaging.AviCompressor)cboVideoCodec.SelectedItem;

            // Enable or disable the buttons.
            cmdShowPropertyPage.Enabled = _selectedCodec.PropertyPageAvailable;
            cmdLoadData.Enabled = _selectedCodec.PropertyPageAvailable;
            cmdSaveData.Enabled = _selectedCodec.PropertyPageAvailable;
        }
		//>>

        /// <summary>
        /// cboVideoCodec_SelectedValueChanged
        ///
        /// If the selected codec has a property dialog, the buttons
        /// will be enabled.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<cboVideoCodec_Click
        private void cboVideoCodec_SelectedIndexChanged(object sender, EventArgs e)
        {
            _selectedCodec = (TIS.Imaging.AviCompressor)cboVideoCodec.SelectedItem;
            // Enable or disable the buttons.
            cmdShowPropertyPage.Enabled = _selectedCodec.PropertyPageAvailable;
            cmdLoadData.Enabled = _selectedCodec.PropertyPageAvailable;
            cmdSaveData.Enabled = _selectedCodec.PropertyPageAvailable;
        }
		//>>

        /// <summary>
        /// cmdShowPropertyPage_Click
        ///
        /// Shows the property dialog of a codec by calling its
        /// ShowPropertyPage Method.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<cmdShowPropertyPage_Click
        private void cmdShowPropertyPage_Click(object sender, EventArgs e)
        {
            _selectedCodec.ShowPropertyPage();
        }
		//>>

        /// <summary>
        /// cmdSaveData_Click
        ///
        /// Gets the binary data from the codec and saves it
        /// into the binary opened file "test.bin".
        /// To make sure that the saved file will match the used
        /// codec, the name of the codec will be saved in the file.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<SaveData
        private void cmdSaveData_Click(object sender, EventArgs e)
        {
            try
            {
                System.IO.FileStream filestream = new System.IO.FileStream("test.bin", System.IO.FileMode.Create, System.IO.FileAccess.Write);
                System.IO.BinaryWriter binWriter = new System.IO.BinaryWriter(filestream);
                binWriter.Write(_selectedCodec.Name);
                binWriter.Write(_selectedCodec.CompressorDataSize);
                binWriter.Write(_selectedCodec.CompressorData);

                binWriter.Close();
                filestream.Close();
            }
            catch (Exception Ex)
            {
                MessageBox.Show(Ex.Message);
            }
        }
		//>>

        /// <summary>
        /// cmdLoadData_Click
        ///
        /// Loads binary data from a file "test.bin" and assigns
        /// it to the codec
        /// To check, whether the file matches the used codec, the
        /// name of the codec was saved in the file. Now, it will be
        /// loaded first from the file and compared with Codec.Name.
        /// If they are identical, the binary data can be assigned
        /// to the codec. Please refer to cmdSaveData_Click().
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<LoadData
        private void cmdLoadData_Click(object sender, EventArgs e)
        {
            try
            {
                System.IO.FileStream filestream = new System.IO.FileStream("test.bin", System.IO.FileMode.Open, System.IO.FileAccess.Read);
                System.IO.BinaryReader binReader = new System.IO.BinaryReader(filestream);

                // Retrieve the name of the codec from the codec configuration file
                string codecName = binReader.ReadString();

                //Compare the codec name in the file with the current codec's name.
                if (_selectedCodec.Name == codecName)
                {
                    // Read the length of the binary data.
                    int codecDataLen = binReader.ReadInt32();
                    // Assign the configuration data to the codec.
                    _selectedCodec.CompressorData = binReader.ReadBytes(codecDataLen);
                }
                else
                {
                    MessageBox.Show("The saved data does not match to the used codec.\n" +
                            "saved: " + codecName + "\n" +
                            "used: " + _selectedCodec.Name);
                }
                binReader.Close();
                filestream.Close();
            }
            catch (Exception Ex)
            {
                MessageBox.Show(Ex.Message);
            }
        }
		//>>

    }
}