using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Capturing_an_AVI_File
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }
            cmdStartLive.Enabled = true;
            cmdStopLive.Enabled = false;
        }

        /// <summary>
        /// cmdStartLive_Click
        ///
        /// Start the live video.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<startlive
        private void cmdStartLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStart();
            cmdStartLive.Enabled = false;
            cmdStopLive.Enabled = true;
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
        }
        //>>

        /// <summary>
        /// cmdCaptureAVI_Click
        ///
        /// Show the AVI Capture dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmdCaptureAVI_Click(object sender, EventArgs e)
        {
            WriteAvi WriteAviDlg = new WriteAvi(icImagingControl1);
            WriteAviDlg.ShowDialog();
            WriteAviDlg.Dispose();
        }


    }
}