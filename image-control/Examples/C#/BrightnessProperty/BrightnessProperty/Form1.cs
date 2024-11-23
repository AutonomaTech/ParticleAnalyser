using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace BrightnessProperty
{
    public partial class Form1 : Form
    {
        // These variables will hold the interfaces to the brightness property
        //<<declare
        private TIS.Imaging.VCDRangeProperty _brightnessRange;
        private TIS.Imaging.VCDSwitchProperty _brightnessSwitch;
        //>>

        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Form1_Load
        ///
        /// Check whether a device has been specified in the properties of IC Imaging
        /// Control. If there is no device, show the device selection dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load( object sender, EventArgs e )
        {
            sldBrightness.Enabled = false;
            chkBrightnessAuto.Enabled = false;

            // Check whether a valid video capture device has been selected,
            // otherwise show the device settings dialog
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            if (_brightnessRange != null)
            {
                //<<setupcontrols
                // Initialize the slider with the current range and value of the
                // BrightnessRange object.
                sldBrightness.Enabled = true;
                sldBrightness.Minimum = _brightnessRange.RangeMin;
                sldBrightness.Maximum = _brightnessRange.RangeMax;
                sldBrightness.Value = _brightnessRange.Value;

                // Initialize the checkbox with the BrightnessSwitch object
                if( _brightnessSwitch != null )
                {
                    chkBrightnessAuto.Enabled = true;
                    sldBrightness.Enabled = !_brightnessSwitch.Switch;
                    chkBrightnessAuto.Checked = _brightnessSwitch.Switch;
                }
                icImagingControl1.LiveStart();
                //>>
            }
        }

        /// <summary>
        /// InitBrightnessItem
        ///
        /// Retrieve the brightness VCDPropertyItem and assign BrightnessSwitch and BrightnessRange.
        /// The function returns true, if the property exists. If the property does not
        /// exists, the function returns false.
        /// </summary>
        /// <returns></returns>
        private void InitBrightnessItem()
        {
            // Try Find brightness property in the VCDPropertyItems collection.
            // If brightness is not support by the current video capture device,
            // an exception is thrown.
            //<<retrieve
            TIS.Imaging.VCDPropertyItem brightness = icImagingControl1.VCDPropertyItems.FindItem( TIS.Imaging.VCDGUIDs.VCDID_Brightness );
            //>>
            if( brightness != null )
            {
                //<<getswitchandrange
                // Acquire interfaces to the range and switch interface for value and auto
                _brightnessRange = brightness.Find<TIS.Imaging.VCDRangeProperty>( TIS.Imaging.VCDGUIDs.VCDElement_Value );
                _brightnessSwitch = brightness.Find<TIS.Imaging.VCDSwitchProperty>( TIS.Imaging.VCDGUIDs.VCDElement_Auto );
                if( _brightnessSwitch == null )
                {
                    MessageBox.Show( "Automation of brightness is not supported by the current device!" );
                }
                //>>
            }

            // Show a message box, if brightness is not supported.
            if( brightness == null )
            {
                MessageBox.Show( "Brightness property is not supported by the current device!" );
            }
        }

        /// <summary>
        /// chkBrightnessAuto_CheckedChanged
        ///
        /// Enable or disable the automatic of the brightness property when the checkbox
        /// has been clicked.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<check
        private void chkBrightnessAuto_CheckedChanged( object sender, EventArgs e )
        {
            _brightnessSwitch.Switch = chkBrightnessAuto.Checked;
            sldBrightness.Enabled = !chkBrightnessAuto.Checked;
        }
        //>>

        /// <summary>
        /// sldBrightness_Scroll
        ///
        /// Set the brightness to the current slider position.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<scroll
        private void sldBrightness_Scroll( object sender, EventArgs e )
        {
            _brightnessRange.Value = sldBrightness.Value;
        }
        //>>
    }
}