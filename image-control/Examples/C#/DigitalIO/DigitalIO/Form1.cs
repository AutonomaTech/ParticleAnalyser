using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;
using TIS.Imaging.VCDHelpers;

namespace DigitalIO
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private VCDRangeProperty _gpioIn;
        private VCDRangeProperty _gpioOut;
        private VCDButtonProperty _gpioRead;
        private VCDButtonProperty _gpioWrite;

        //<<formload
        private void Form1_Load( object sender, EventArgs e )
        {
			if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            // Initialize the VCDProp class to access the properties of our ICImagingControl
            // object
            _gpioIn = icImagingControl1.VCDPropertyItems.Find<VCDRangeProperty>( VCDGUIDs.VCDID_GPIO, VCDGUIDs.VCDElement_GPIOIn );
            _gpioOut = icImagingControl1.VCDPropertyItems.Find<VCDRangeProperty>( VCDGUIDs.VCDID_GPIO, VCDGUIDs.VCDElement_GPIOOut );
            _gpioRead = icImagingControl1.VCDPropertyItems.Find<VCDButtonProperty>( VCDGUIDs.VCDID_GPIO, VCDGUIDs.VCDElement_GPIORead );
            _gpioWrite = icImagingControl1.VCDPropertyItems.Find<VCDButtonProperty>( VCDGUIDs.VCDID_GPIO, VCDGUIDs.VCDElement_GPIOWrite );

            // First of all, check, whether the digital IOs are supported by the
            // current video capture device.
            if( _gpioIn != null && _gpioOut != null && _gpioRead != null && _gpioWrite != null )
            {
                // Get the digital input state.
                cmdReadDigitalInput.Enabled = true;
                ReadDigitalInput();

                // Get the digital output state.
                cmdWriteDigitalOutput.Enabled = true;
                chkDigitalOutputState.Enabled = true;
                if( _gpioOut.Value == 1 )
                {
                    chkDigitalOutputState.CheckState = CheckState.Checked;
                }
                else
                {
                    chkDigitalOutputState.CheckState = CheckState.Unchecked;
                }
            }
            else
            {
                cmdReadDigitalInput.Enabled = false;
                cmdWriteDigitalOutput.Enabled = false;
                chkDigitalOutputState.Enabled = false;
            }

            // Initialize the sliders
            // start live mode
            icImagingControl1.LiveStart();
        }
        //>>

        /// <summary>
        /// ReadDigitalInput
        ///
        /// Send the push for read out to the video capture device. Then set the
        /// input state check box to the current state of the digital input.
        /// </summary>
        private void ReadDigitalInput()
        {
            // Read the digital input from the video capture device.
            _gpioRead.Push();

            // Set the state of the digital input to the chkDigitalInputState
            // check box.
            if( _gpioIn.Value == 1 )
            {
                chkDigitalInputState.CheckState = CheckState.Checked;
            }
            else
            {
                chkDigitalInputState.CheckState = CheckState.Unchecked;
            }
        }

        private void cmdReadDigitalInput_Click( object sender, EventArgs e )
        {
            ReadDigitalInput();
        }

        /// <summary>
        /// cmdWriteDigitalOutput_Click
        ///
        /// The state of the chkDigitalOutputState check box is set to the video 
        /// capture device' digital output property. 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmdWriteDigitalOutput_Click( object sender, EventArgs e )
        {
            // Set the state.
            if( chkDigitalOutputState.CheckState == CheckState.Checked )
            {
                _gpioOut.Value = 1;
            }
            else
            {
                _gpioOut.Value = 0;
            }

            // Now write it into the video capture device.
            _gpioWrite.Push();
        }
    }
}