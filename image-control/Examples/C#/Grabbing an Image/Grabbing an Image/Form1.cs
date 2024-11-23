using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;

namespace Grabbing_an_Image
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Form1_Load
        ///
        /// If no device has been selected in the properties window of IC Imaging
        /// Control, the device settings dialog of IC Imaging Control is show at
        /// start of this sample. 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", "Grabbing an Image", MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
            }

            cmdStartLive.Enabled = true;
            cmdStopLive.Enabled = false;
            cmdSaveBitmap.Enabled = false;
        }

        /// <summary>
        /// cmdStartLive_Click
        ///
        /// Start the live video. A valid video capture device should have been
        /// selected previsously in the properties window of IC Imaging Control.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<startlive
        private void cmdStartLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.Sink = new TIS.Imaging.FrameSnapSink();

            icImagingControl1.LiveStart();

            cmdStartLive.Enabled = false;
            cmdStopLive.Enabled = true;
            cmdSaveBitmap.Enabled = true;
        }
        //>>
        /// <summary>
        /// cmdStopLive_Click
        ///
        /// Stop the live video.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<stoplive
        private void cmdStopLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStop();

            cmdStartLive.Enabled = true;
            cmdStopLive.Enabled = false;
            cmdSaveBitmap.Enabled = false;
        }
        //>>

        /// <summary>
        /// cmdSaveBitmap_Click
        ///
        /// Snap an image from the live video stream and show the file save
        /// dialog. After a file name has been selected, the image is saved.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<savebitmap
        private void cmdSaveBitmap_Click( object sender, EventArgs e )
        {
            TIS.Imaging.FrameSnapSink snapSink = icImagingControl1.Sink as TIS.Imaging.FrameSnapSink;

            TIS.Imaging.IFrameQueueBuffer frm = snapSink.SnapSingle(TimeSpan.FromSeconds(5));
            
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();
            saveFileDialog1.Filter = "bmp files (*.bmp)|*.bmp|All files (*.*)|*.*";
            saveFileDialog1.FilterIndex = 1;
            saveFileDialog1.RestoreDirectory = true;

            if( saveFileDialog1.ShowDialog() == DialogResult.OK )
            {
                frm.SaveAsBitmap(saveFileDialog1.FileName);
            }
        }
		//>>
    }
}