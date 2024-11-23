using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

using TIS.Imaging;

namespace demoapp
{
    public partial class AviDialog : Form
    {
        // AviDialog
        //
        // This dialog implements all controls for codec select and AVI capture
        //
        public TIS.Imaging.ICImagingControl _ImagingControl;
        private BaseSink _savedSink;

        public AviDialog( TIS.Imaging.ICImagingControl ic )
            : base()
        {

            //This call is required by the Windows Form Designer.
            InitializeComponent();

            //Add any initialization after the InitializeComponent() call
            _ImagingControl = ic;

        }

        //
        // AviDialog_Load
        //
        // Setup the controls. From the registry the last used codec, codec configuration
        // file name and AVI file name is retrieved. The related controls are filled with
        // these items.
        // In the file DeviceSettings.vb are the registry functions implemented.
        //
        private void AviDialog_Load( object sender, System.EventArgs e )
        {
            cmdSaveCodecData.Enabled = false;
            cmdLoadCodecData.Enabled = false;
            cmdShowPropPage.Enabled = false;
            cmdStopButton.Enabled = false;
            cmdStartButton.Enabled = true;

            //Insert all installed codecs into the CodecBox combobox.
            CodecBox.DataSource = AviCompressor.AviCompressors ;

            string LastUsedCodec = DeviceSettings.GetCodecFromRegisty(); // Retrieve the name of the last used codec.
            // Select the last used codec in the codec combobox and select it.
            CodecBox.SelectedIndex = 0;

            for( int i = 0; i < CodecBox.Items.Count; i++ )
            {
                if( ((AviCompressor)(CodecBox.Items[i])).Name == LastUsedCodec )
                {
                    CodecBox.SelectedIndex = i;
                    break;
                }
            }

            // Get the last used file names from the registry
            edtConfigFile.Text = DeviceSettings.GetCodecFileFromRegisty();
            edtAVIFile.Text = DeviceSettings.GetAVIFileFromRegisty();
        }

        //
        // CodecBox_Changed
        //
        // Handler sub for the selection change event of the codec combo box.
        // It is checked, whether the new selected codec has a configuration dialog
        // and is able to export and import configuration data. Depending on these
        // capabilities the appropriate controls of the AVI Dialog will be enabled or
        // disabled.
        //
        private void CodecBox_Changed( object sender, System.EventArgs e )
        {
            AviCompressor Codec = ((AviCompressor)(CodecBox.SelectedItem));

            // Check for the configuration dialog
            if( Codec.PropertyPageAvailable )
            {
                cmdShowPropPage.Enabled = true;
            }
            else
            {
                cmdShowPropPage.Enabled = false;
            }

            // Check, whether the codec exports and imports configuration data
            if( Codec.CompressorDataSize > 0 )
            {
                cmdSaveCodecData.Enabled = true;
                cmdLoadCodecData.Enabled = true;
                edtConfigFile.Enabled = true;
                cmdCodecDataFileChooser.Enabled = true;
            }
            else
            {
                cmdSaveCodecData.Enabled = false;
                cmdLoadCodecData.Enabled = false;
                edtConfigFile.Enabled = false;
                cmdCodecDataFileChooser.Enabled = false;
            }
        }

        //
        // cmdAVIFileChooser_Click
        //
        // Click event handler for the cmdAVIFileChooser. Opens a file selection box.
        //
        private void cmdAVIFileChooser_Click( object sender, System.EventArgs e )
        {
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();

            saveFileDialog1.Filter = "avi files (*.avi)|*.avi|All files (*.*)|*.*";
            saveFileDialog1.FilterIndex = 1;
            saveFileDialog1.RestoreDirectory = true;

            if( saveFileDialog1.ShowDialog() == DialogResult.OK )
            {
                edtAVIFile.Text = saveFileDialog1.FileName;
            }
        }

        //
        // cmdStartButton_Click
        //
        // Start the AVI capturing.
        //
        //<<StartCapture
        private void cmdStartButton_Click( object sender, System.EventArgs e )
        {
            if( _ImagingControl.DeviceValid )
            {
                _savedSink = _ImagingControl.Sink;
                _ImagingControl.Sink = new MediaStreamSink( (AviCompressor)CodecBox.SelectedItem, edtAVIFile.Text );
                _ImagingControl.LiveStart();
                cmdStartButton.Enabled = false;
                cmdStopButton.Enabled = true;
            }
        }
        //>>
        //
        // cmdStopButton_Click
        //
        // Stop the AVI capturing.
        //
        private void cmdStopButton_Click( object sender, System.EventArgs e )
        {
            if( _ImagingControl.DeviceValid )
            {
                _ImagingControl.LiveStop();
                _ImagingControl.Sink = _savedSink;
                _savedSink = null;
                cmdStartButton.Enabled = true;
                cmdStopButton.Enabled = false;
            }
        }

        //
        // chkPause_Click
        //
        // Pause or restart the avi capture according to the value in chkPause.
        //
        //<<chkpauseclick
        private void chkCapturePause_CheckedChanged( object sender, System.EventArgs e )
        {
            _ImagingControl.LivePause = chkCapturePause.Checked;
        }
        //>>

        //
        // cmdShowPropPage_Click
        //
        // Event handler sub of the codec property button. If the current
        // seleceted codec has a property dialog, the property button is
        // enabled. A click on this button opens the codec's property dialog.
        //
        private void cmdShowPropPage_Click( object sender, System.EventArgs e )
        {
            AviCompressor Codec = null;
            Codec = ((AviCompressor)(CodecBox.SelectedItem));
            Codec.ShowPropertyPage();
        }

        //
        // cmdCloseButton_Click
        //
        // Event handler sub of the close button.
        //
        private void cmdCloseButton_Click( object sender, System.EventArgs e )
        {
            Close();
        }

        //
        // cmdCodecDataFileChooser_Click
        //
        // Click event handler for the cmdCodecDataFileChooser button. Opens a file selection box.
        //
        private void cmdCodecDataFileChooser_Click( object sender, System.EventArgs e )
        {
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();

            saveFileDialog1.Filter = "codec data files (*.cod)|*.cod|All files (*.*)|*.*";
            saveFileDialog1.FilterIndex = 1;
            saveFileDialog1.RestoreDirectory = true;

            if( saveFileDialog1.ShowDialog() == DialogResult.OK )
            {
                edtConfigFile.Text = saveFileDialog1.FileName;
            }
        }


        //
        // cmdSaveCodecData_Click
        //
        // Save the configuration properties of the current selected codec to a
        // binary file.
        //
        //<<SaveData
        private void cmdSaveCodecData_Click( object sender, System.EventArgs e )
        {
            if( edtConfigFile.Text != "" )
            {
                try
                {
                    System.IO.FileStream Filestream = new System.IO.FileStream( edtConfigFile.Text, System.IO.FileMode.Create, System.IO.FileAccess.Write );

                    // Create the writer for data.
                    System.IO.BinaryWriter BinWriter = new System.IO.BinaryWriter( Filestream );

                    // Write data to Test.data.
                    AviCompressor aviComp = null;
                    aviComp = ((AviCompressor)(CodecBox.SelectedItem));

                    BinWriter.Write( aviComp.ToString() );
                    BinWriter.Write( aviComp.CompressorDataSize );
                    BinWriter.Write( aviComp.CompressorData );

                    BinWriter.Close();
                    Filestream.Close();

                }
                catch( Exception Ex )
                {
                    MessageBox.Show( Ex.Message, "Write error", MessageBoxButtons.OK, MessageBoxIcon.Error );
                }
            }
        }
        //>>

        //
        // cmdLoadCodecData_Click
        //
        // The previously saved codec configuration is loaded from the
        // file specified in edtConfigFile.Text. The codec configuration
        // contains the name of the codec. This name is searched in the
        // codec combobox. If it was found, the codec is selected and the
        // configuration data is assigned to this codec.
        //
        //<<LoadData
        private void cmdLoadCodecData_Click( object sender, System.EventArgs e )
        {
            if( edtConfigFile.Text != "" )
            {
                try
                {
                    // Create the reader for data.
                    System.IO.FileStream Filestrem = new System.IO.FileStream( edtConfigFile.Text, System.IO.FileMode.Open, System.IO.FileAccess.Read );
                    System.IO.BinaryReader BinReader = new System.IO.BinaryReader( Filestrem );

                    string CodecName = "";
                    AviCompressor Codec = null;
                    bool CodecFound = false;

                    // Retrieve the name of the codec from the codec configuration file
                    CodecName = BinReader.ReadString();

                    // Search this codec in the codec combo box
                    CodecFound = false;
                    foreach( AviCompressor item in CodecBox.Items )
                    {
                        if( item.ToString() == CodecName )
                        {
                            CodecBox.SelectedItem = item;
                            Codec = item;
                            CodecFound = true;
                        }
                    }

                    if( CodecFound == true )
                    {
                        int codecDataLen = BinReader.ReadInt32();
                        // Assign the configuration data to the codec.
                        Codec.CompressorData = BinReader.ReadBytes( codecDataLen );
                    }
                    else
                    {
                        // If the codec was not found, display an error message.
                        MessageBox.Show( "The codec " + CodecName + " was not found!", "Load codec configuration", MessageBoxButtons.OK, MessageBoxIcon.Information );
                    }
                    BinReader.Close();
                    Filestrem.Close();

                }
                catch( Exception Ex )
                {
                    MessageBox.Show( Ex.Message, "Read error", MessageBoxButtons.OK, MessageBoxIcon.Error );
                }
            }
        }
        //>>

        //
        // AviDialog_Closing
        //
        // When the dialog is closed, the contents of the controls is save to the
        // registry. This contents will be loaded from the registry, when the 
        // AVI dialog is opened again.
        //
        private void AviDialog_Closing( object sender, System.ComponentModel.CancelEventArgs e )
        {
            DeviceSettings.SaveAVIFileInRegisty( edtAVIFile.Text );
            DeviceSettings.SaveCodecInRegisty( ((AviCompressor)(CodecBox.SelectedItem)).Name );
            DeviceSettings.SaveCodecFileInRegisty( edtConfigFile.Text );
        }
    }
}