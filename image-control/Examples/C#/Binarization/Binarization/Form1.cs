using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Binarization
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private TIS.Imaging.FrameFilter _frameFilter;

        private void Form1_Load( object sender, EventArgs e )
        {
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            //<<initgrabber
            // Disable overlay bitmap
            icImagingControl1.OverlayBitmapPosition = TIS.Imaging.PathPositions.None;

            // Create an instance of the frame filter implementation
            BinarizationFilter binFilterImpl = new BinarizationFilter();

            // Create a FrameFilter object wrapping the implementation
            _frameFilter = TIS.Imaging.FrameFilter.Create( binFilterImpl );

            // Set the FrameFilter as display frame filter.
            icImagingControl1.DisplayFrameFilters.Add( _frameFilter );

            // Start live mode
            icImagingControl1.LiveStart();
            //>>

            _frameFilter.BeginParameterTransfer();

            chkEnable.Checked = _frameFilter.GetBoolParameter( "enable" );

            sldThreshold.Minimum = 0;
            sldThreshold.Maximum = 255;
            sldThreshold.Value = _frameFilter.GetIntParameter( "threshold" );
            lblThreshold.Text = sldThreshold.Value.ToString();
            sldThreshold.Enabled = chkEnable.Checked;
            lblThreshold.Enabled = chkEnable.Checked;

            _frameFilter.EndParameterTransfer();
        }

        /// <summary>
        /// The user clicked the "Enable" check box.
		///
		///	Toggle binarization in the binarization filter,
        ///	and adjust the enabled state of the threshold slider.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<check
        private void chkEnable_CheckedChanged( object sender, EventArgs e )
        {
            _frameFilter.BeginParameterTransfer();

            _frameFilter.SetBoolParameter( "enable", chkEnable.Checked );
            sldThreshold.Enabled = _frameFilter.GetBoolParameter( "enable" );
            lblThreshold.Enabled = _frameFilter.GetBoolParameter( "enable" );

            _frameFilter.EndParameterTransfer();
        }
        //>>

        /// <summary>
        /// The user changed the position of the threshold slider.
        ///
        ///	Read the new value, and set it at the binarization filter.
        ///	Display the value in the threshold static control.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<hscroll
        private void sldThreshold_Scroll( object sender, EventArgs e )
        {
            _frameFilter.BeginParameterTransfer();

            _frameFilter.SetIntParameter( "threshold", sldThreshold.Value );
            lblThreshold.Text = _frameFilter.GetIntParameter( "threshold" ).ToString();

            _frameFilter.EndParameterTransfer();
        }
        //>>

        /// <summary>
        /// The user clicked the "Device..."-button
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnDevice_Click( object sender, EventArgs e )
        {
            icImagingControl1.LiveStop();
            icImagingControl1.ShowDeviceSettingsDialog();
            icImagingControl1.LiveStart();
        }

        /// <summary>
        /// The user clicked the "Properties..."-button
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnProperties_Click( object sender, EventArgs e )
        {
            icImagingControl1.ShowPropertyDialog();
        }

        private void Form1_FormClosing( object sender, FormClosingEventArgs e )
        {
            icImagingControl1.LiveStop();
        }

    }
}