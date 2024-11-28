using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;
using TIS.Imaging.VCDHelpers;

namespace Strobe
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        //<<globals
        // Declare the interface reference here, if will later be filled in form load.
        VCDSwitchProperty _strobeEnable;
        //>>

        //<<formload
        private void Form1_Load( object sender, EventArgs e )
        {
            // Check whether a valid video capture device has been selected,
            // otherwise show the device settings dialog
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            // initialize the Interface we will use to enable/disable strobe
            _strobeEnable = icImagingControl1.VCDPropertyItems.Find<VCDSwitchProperty>( VCDGUIDs.VCDID_Strobe, VCDGUIDs.VCDElement_Value );

            // Initialize the sliders
            if( _strobeEnable == null )
            {
                chkStrobe.Enabled = false;
            }
            else
            {
                chkStrobe.Enabled = true;
                // Set the strobe checkbox to the current state to the strobe in
                // the video capture device.
                if( _strobeEnable.Switch == true )
                {
                    chkStrobe.CheckState = CheckState.Checked;
                }
                else
                {
                    chkStrobe.CheckState = CheckState.Unchecked;
                }
            }

            // start live mode
            icImagingControl1.LiveStart();
        }
        //>>

        /// <summary>
        /// chkStrobe_CheckedChanged
        ///
        /// If the user clicks the strobe checkbox, the strobe of the video capture
        /// device is enabled or disabled regarding to the current state of the
        /// the check box. The strobe uses the "Switch" interface.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void chkStrobe_CheckedChanged( object sender, EventArgs e )
        {
            if( chkStrobe.CheckState == CheckState.Checked )
            {
                _strobeEnable.Switch = true;
            }
            else
            {
                _strobeEnable.Switch = false;
            }
        }



    }
}