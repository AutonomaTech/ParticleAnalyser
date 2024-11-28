using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Scroll_And_Zoom
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load( object sender, EventArgs e )
        {
            cmdStart.Enabled = false;
            cmdStop.Enabled = false;
            cmdImageSettings.Enabled = false;
            chkDisplayDefault.Enabled = false;
            chkScrollbarsEnable.Enabled = false;
            sldZoom.Enabled = false;
            lblZoom.Enabled = false;
        }

        //<<cmddevice
        private void cmdDevice_Click(object sender, EventArgs e)
        {
            if (icImagingControl1.DeviceValid)
            {
                if (icImagingControl1.LiveVideoRunning)
                {
                    icImagingControl1.LiveStop();
                }
            }
            icImagingControl1.ShowDeviceSettingsDialog();
            if (icImagingControl1.DeviceValid)
            {
                cmdStart.Enabled = true;
                cmdStop.Enabled = true;
                cmdImageSettings.Enabled = true;
                chkDisplayDefault.Enabled = true;
                chkScrollbarsEnable.Enabled = true;
                sldZoom.Enabled = true;
                lblZoom.Enabled = true;
                chkDisplayDefault.Checked = false;

                icImagingControl1.LiveDisplayDefault = false;
                icImagingControl1.LiveDisplaySize = icImagingControl1.VideoFormatCurrent.Size;

                chkScrollbarsEnable.Checked = icImagingControl1.ScrollbarsEnabled;

                // Enable or disable the slider for the zoom factor, depending
                // on the LiveDisplayDefault property.
                sldZoom.Enabled = !icImagingControl1.LiveDisplayDefault;
                sldZoom.Value = (int)(icImagingControl1.LiveDisplayZoomFactor * 10);
                lblZoomPercent.Text = (sldZoom.Value * 10).ToString() + "%";
            }
        }
//>>

        /// <summary>
        /// sldZoom_Scroll
        ///
        /// Set a new zoom factor.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
//<<sldZoom
        private void sldZoom_Scroll(object sender, EventArgs e)
        {
            if (icImagingControl1.LiveDisplayDefault == false)
            {
                icImagingControl1.LiveDisplayZoomFactor = (float)sldZoom.Value / 10.0f;
                lblZoomPercent.Text = (sldZoom.Value * 10).ToString() + "%";
            }
            else
            {
                MessageBox.Show("The zoom factor can only be set" + "\n" + "if LiveDisplayDefault returns False!");
            }
        }
//>>

        /// <summary>
        /// cmdImageSettings_Click
        ///
        /// Show the image settings dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
//<<cmdimagesetting
        private void cmdImageSettings_Click(object sender, EventArgs e)
        {
            icImagingControl1.ShowPropertyDialog();
        }
//>>

        /// <summary>
        /// cmdStart_Click
        /// Start the live video and draw a rectangle around the live video.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
//<<start
        private void cmdStart_Click(object sender, EventArgs e)
        {
			if( icImagingControl1.DeviceValid )
			{
				icImagingControl1.LiveStart();

                var ovb = icImagingControl1.OverlayBitmapAtPath[TIS.Imaging.PathPositions.Device];

                int width = icImagingControl1.VideoFormatCurrent.Width;
                int height = icImagingControl1.VideoFormatCurrent.Height;

                // Draw a rectangle around the whole image to visualize its perimeter.;
                ovb.Enable = true;
				ovb.DrawLine( Color.FromArgb( 255, 0, 0 ), 0, 0, width - 1, 0 );
				ovb.DrawLine( Color.FromArgb( 255, 0, 0 ), width - 1, 0, width - 1, height - 1 );
				ovb.DrawLine( Color.FromArgb( 255, 0, 0 ), width - 1, height - 1, 0, height - 1 );
				ovb.DrawLine( Color.FromArgb( 255, 0, 0 ), 0, 0, 0, height );
				ovb.DrawText( Color.FromArgb( 255, 0, 0 ), 5, 5, "Scroll and Zoom Sample" );
			}
        }
//>>

        /// <summary>
        /// cmdStop_Click
        ///
        /// Stop the live video.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
//<<stop
        private void cmdStop_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStop();
        }
//>>

        /// <summary>
        /// chkDisplayDefault_CheckedChanged
        ///
        /// Enable or disable the LiveDisplayDefault property.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
//<<chkDisplayDefault_Click
        private void chkDisplayDefault_CheckedChanged(object sender, EventArgs e)
        {
            icImagingControl1.LiveDisplayDefault = chkDisplayDefault.Checked;
            sldZoom.Value = (int)(icImagingControl1.LiveDisplayZoomFactor * 10.0f);
            lblZoomPercent.Text = (sldZoom.Value * 10).ToString() + "%";
            sldZoom.Enabled = !chkDisplayDefault.Checked;
        }
//>>

        /// <summary>
        /// chkScrollbarsEnable_CheckedChanged
        ///
        /// Enable or disable the scroll bars.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
//<<chkScrollbarsEnable_Click
        private void chkScrollbarsEnable_CheckedChanged(object sender, EventArgs e)
        {
            icImagingControl1.ScrollbarsEnabled = chkScrollbarsEnable.Checked;
        }
//>>

		/// <summary>
		/// When the user uses a scroll bar to move the live image, update
		/// lblScrollPosition.Text
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
//<<onscroll
		private void icImagingControl1_Scroll( object sender, ScrollEventArgs e )
		{
			Point p = icImagingControl1.AutoScrollPosition;
			lblScrollPosition.Text = string.Format( "{0}/{1}", p.X, p.Y );
		}
        //>>
    }
}