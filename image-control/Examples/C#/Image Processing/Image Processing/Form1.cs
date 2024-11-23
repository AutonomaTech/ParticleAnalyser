using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;

namespace Image_Processing
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Form_load
        ///
        /// Shows the device settings dialog and checks for a  valid video
        /// capture device to prevent error messages.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        /// 
        //<<Form1_Load
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

            cmdStartLive.Enabled = true;
            cmdStopLive.Enabled = false;
            cmdProcess.Enabled = false;
        }
        //>>Form1_Load

        Form _displayForm = null;

        /// <summary>
        /// cmdProcess_Click
        ///
        /// Snaps a single image, inverts the image data and shows the buffer.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<cmd_process
        private void cmdProcess_Click( object sender, EventArgs e )
        {
            Cursor = Cursors.WaitCursor;
            try
            {
                FrameSnapSink sink = icImagingControl1.Sink as FrameSnapSink;

                IFrame imgBuffer = sink.SnapSingle( TimeSpan.FromSeconds( 5 ) );

                var frameType = imgBuffer.FrameType;

                unsafe
                {
                    byte* pDatabyte = imgBuffer.Ptr;
                    for( int y = 0; y < frameType.Height; y++ )
                    {
                        for( int x = 0; x < frameType.BytesPerLine; x++ )
                        {
                            *pDatabyte = (byte)(255 - *pDatabyte);
                            pDatabyte++;
                        }
                    }
                }

                if( _displayForm == null || _displayForm.IsDisposed )
                {
                    _displayForm = new Form();
                }

                _displayForm.BackgroundImage = imgBuffer.CreateBitmapCopy();
                _displayForm.Size = frameType.Size;
                _displayForm.Show();
            }
            catch( Exception ex )
            {
                MessageBox.Show( ex.Message );
            }
            Cursor = Cursors.Default;
        }
        //>>

        //<<cmd_StopLive
        private void cmdStopLive_Click( object sender, EventArgs e )
        {
            icImagingControl1.LiveStop();

            cmdStartLive.Enabled = true;
            cmdStopLive.Enabled = false;
            cmdProcess.Enabled = false;
        }
        //>>

        //<<cmd_StartLive
        private void cmdStartLive_Click_1( object sender, EventArgs e )
        {
            // This sample works works for color images, so set the sink type to RGB24
            icImagingControl1.Sink = new TIS.Imaging.FrameSnapSink(MediaSubtypes.RGB24);

            icImagingControl1.LiveStart();

            cmdStartLive.Enabled = false;
            cmdStopLive.Enabled = true;
            cmdProcess.Enabled = true;
        }
        //>>

    }
}