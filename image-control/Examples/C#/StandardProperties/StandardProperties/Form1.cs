using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;
using TIS.Imaging.Helper;

namespace StandardProperties
{
    public partial class Form1 : Form
    {
        // These variables will hold the interfaces to the properties
        private TIS.Imaging.VCDAbsoluteValueProperty _exposureValue;
        private TIS.Imaging.VCDSwitchProperty _exposureAuto;
        private TIS.Imaging.VCDAbsoluteValueProperty _gainValue;
        private TIS.Imaging.VCDSwitchProperty _gainAuto;


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
            trackBarExposure.Enabled = false;
            checkBoxExposure.Enabled = false;
            trackBarGain.Enabled = false;
            checkBoxGain.Enabled = false;

            // Check whether a valid video capture device has been selected,
            // otherwise show the device settings dialog
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            InitProperties();

            icImagingControl1.LiveStart();
        }

        private void    InitProperties()
        {
            _exposureValue = icImagingControl1.VCDPropertyItems.Find<VCDAbsoluteValueProperty>(VCDGUIDs.VCDID_Exposure, VCDGUIDs.VCDElement_Value);
            _exposureAuto = icImagingControl1.VCDPropertyItems.Find<VCDSwitchProperty>(VCDGUIDs.VCDID_Exposure, VCDGUIDs.VCDElement_Auto);
            _gainValue = icImagingControl1.VCDPropertyItems.Find<VCDAbsoluteValueProperty>(VCDGUIDs.VCDID_Gain, VCDGUIDs.VCDElement_Value);
            _gainAuto = icImagingControl1.VCDPropertyItems.Find<VCDSwitchProperty>(VCDGUIDs.VCDID_Gain, VCDGUIDs.VCDElement_Auto);

            if( _exposureValue != null )
            {
                trackBarExposure.Enabled = _exposureValue.Available && !_exposureValue.ReadOnly;
                trackBarExposure.Minimum = 0;
                trackBarExposure.Maximum = 100;
                trackBarExposure.Value = AbsoluteValueSliderHelper.AbsValToSliderPosLogarithmic(_exposureValue, _exposureValue.Value, 100);
            }
            if( _exposureAuto != null )
            {
                checkBoxExposure.Enabled = !_exposureAuto.ReadOnly;
                checkBoxExposure.Checked = _exposureAuto.Switch;
            }
            if( _gainValue != null )
            {
                trackBarGain.Enabled = _gainValue.Available && !_gainValue.ReadOnly;
                trackBarGain.Minimum = 0;
                trackBarGain.Maximum = 100;
                trackBarGain.Value = AbsoluteValueSliderHelper.AbsValToSliderPosLinear(_gainValue, _gainValue.Value, 100);
            }
            if( _gainAuto != null )
            {
                checkBoxGain.Enabled = !_gainAuto.ReadOnly;
                checkBoxGain.Checked = _gainAuto.Switch;
            }

        }

        private void CheckBoxExposure_CheckedChanged( object sender, EventArgs e )
        {
            _exposureAuto.Switch = checkBoxExposure.Checked;
            trackBarExposure.Enabled = _exposureValue != null && _exposureValue.Available && !_exposureValue.ReadOnly;
        }

        private void TrackBarExposure_Scroll( object sender, EventArgs e )
        {
            _exposureValue.Value = AbsoluteValueSliderHelper.SliderPosToAbsValLogarithmic( _exposureValue, trackBarExposure.Value, 100 );
        }

        private void CheckBoxGain_CheckedChanged( object sender, EventArgs e )
        {
            _gainAuto.Switch = checkBoxGain.Checked;
            trackBarGain.Enabled = _gainValue != null && _gainValue.Available && !_gainValue.ReadOnly;
        }

        private void TrackBarGain_Scroll( object sender, EventArgs e )
        {
            _gainValue.Value = AbsoluteValueSliderHelper.SliderPosToAbsValLinear(_gainValue, trackBarGain.Value, 100);
        }
    }
}