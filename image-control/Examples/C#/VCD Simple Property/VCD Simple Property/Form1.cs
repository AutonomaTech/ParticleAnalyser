using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;

namespace VCD_Simple_Property
{
    public partial class Form1 : Form
    {
        //<<globals
        private TIS.Imaging.VCDSwitchProperty _brightnessAuto;
        private TIS.Imaging.VCDSwitchProperty _whitebalanceAuto;
        private TIS.Imaging.VCDRangeProperty _brightnessRange;
        private TIS.Imaging.VCDRangeProperty _whitebalanceBlueRange;
        private TIS.Imaging.VCDRangeProperty _whitebalanceRedRange;
        //>>
        public Form1()
        {
            InitializeComponent();
        }

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

            // Initialize the auto check boxes
            _brightnessAuto = icImagingControl1.VCDPropertyItems.Find<VCDSwitchProperty>( VCDGUIDs.VCDID_Brightness, VCDGUIDs.VCDElement_Auto );
            if( _brightnessAuto == null )
            {
                BrightnessAutoCheckBox.Enabled = false;
            }
            else
            {
                _brightnessAuto.Switch = false;
            }

            _whitebalanceAuto = icImagingControl1.VCDPropertyItems.Find<VCDSwitchProperty>( VCDGUIDs.VCDID_WhiteBalance, VCDGUIDs.VCDElement_Auto );
            if( _whitebalanceAuto == null )
            {
                WhitebalanceCheckBox.Enabled = false;
                WhitebalanceOnePushButton.Enabled = false;
            }
            else
            {
                _whitebalanceAuto.Switch = false;
            }


            // Initialize the sliders
            _brightnessRange = icImagingControl1.VCDPropertyItems.Find<VCDRangeProperty>( VCDGUIDs.VCDID_Brightness, VCDGUIDs.VCDElement_Value );
            if( _brightnessRange == null )
            {
                BrightnessTrackBar.Enabled = false;
            }
            else
            {
                BrightnessTrackBar.Enabled = true;
                BrightnessTrackBar.Minimum = _brightnessRange.RangeMin;
                BrightnessTrackBar.Maximum = _brightnessRange.RangeMax;
                BrightnessTrackBar.Value = _brightnessRange.Value;
                BrightnessTrackBar.TickFrequency = (BrightnessTrackBar.Maximum - BrightnessTrackBar.Minimum) / 10;
                BrightnessValueLabel.Text = BrightnessTrackBar.Value.ToString();
            }

            _whitebalanceBlueRange = icImagingControl1.VCDPropertyItems.Find<VCDRangeProperty>( VCDGUIDs.VCDID_WhiteBalance, VCDGUIDs.VCDElement_WhiteBalanceBlue );
            if( _whitebalanceBlueRange == null )
            {
                WhiteBalBlueTrackBar.Enabled = false;
            }
            else
            {
                WhiteBalBlueTrackBar.Enabled = true;
                WhiteBalBlueTrackBar.Minimum = _whitebalanceBlueRange.RangeMin;
                WhiteBalBlueTrackBar.Maximum = _whitebalanceBlueRange.RangeMax;
                WhiteBalBlueTrackBar.Value = _whitebalanceBlueRange.Value;
                WhiteBalBlueTrackBar.TickFrequency = (WhiteBalBlueTrackBar.Maximum - WhiteBalBlueTrackBar.Minimum) / 10;
                WhiteBalBlueLabel.Text = WhiteBalBlueTrackBar.Value.ToString();
            }

            _whitebalanceRedRange = icImagingControl1.VCDPropertyItems.Find<VCDRangeProperty>( VCDGUIDs.VCDID_WhiteBalance, VCDGUIDs.VCDElement_WhiteBalanceRed );
            if( _whitebalanceRedRange == null )
            {
                WhiteBalRedTrackBar.Enabled = false;
            }
            else
            {
                WhiteBalRedTrackBar.Enabled = false;
                WhiteBalRedTrackBar.Enabled = true;
                WhiteBalRedTrackBar.Minimum = _whitebalanceRedRange.RangeMin;
                WhiteBalRedTrackBar.Maximum = _whitebalanceRedRange.RangeMax;
                WhiteBalRedTrackBar.Value = _whitebalanceRedRange.Value;
                WhiteBalRedTrackBar.TickFrequency = (WhiteBalRedTrackBar.Maximum - WhiteBalRedTrackBar.Minimum) / 10;
                WhiteBalRedLabel.Text = WhiteBalRedTrackBar.Value.ToString();
            }

            // start live mode
            icImagingControl1.LiveStart();
        }
        //>>

        //<<checkbox_brightness
        private void BrightnessAutoCheckBox_CheckedChanged( object sender, EventArgs e )
        {
            _brightnessAuto.Switch = BrightnessAutoCheckBox.Checked;
            BrightnessTrackBar.Enabled = !BrightnessAutoCheckBox.Checked;
        }
        //>>

        //<<checkbox_whitebalance
        private void WhitebalanceCheckBox_CheckedChanged( object sender, EventArgs e )
        {
            _whitebalanceAuto.Switch = WhitebalanceCheckBox.Checked;
            WhiteBalBlueTrackBar.Enabled = !WhitebalanceCheckBox.Checked;
            WhiteBalRedTrackBar.Enabled = !WhitebalanceCheckBox.Checked;
        }
        //>>

        //<<brightness_scroll
        private void BrightnessTrackBar_Scroll( object sender, EventArgs e )
        {
            _brightnessRange.Value = BrightnessTrackBar.Value;
            BrightnessValueLabel.Text = _brightnessRange.Value.ToString();
        }
        //>>

        private void WhiteBalRedTrackBar_Scroll( object sender, EventArgs e )
        {
            _whitebalanceRedRange.Value = WhiteBalRedTrackBar.Value;
            WhiteBalRedLabel.Text = _whitebalanceRedRange.Value.ToString();
        }

        private void WhiteBalBlueTrackBar_Scroll( object sender, EventArgs e )
        {
            _whitebalanceBlueRange.Value = WhiteBalBlueTrackBar.Value;
            WhiteBalBlueLabel.Text = _whitebalanceBlueRange.Value.ToString();
        }

        //<<whitebalance_onepush
        private void WhitebalanceOnePushButton_Click( object sender, EventArgs e )
        {
            var whiteBalanceOnePush = icImagingControl1.VCDPropertyItems.Find<VCDButtonProperty>( VCDGUIDs.VCDID_WhiteBalance, VCDGUIDs.VCDElement_OnePush );
            if( whiteBalanceOnePush != null )
            {
                whiteBalanceOnePush.Push();
            }
        }
        //>>

    }
}