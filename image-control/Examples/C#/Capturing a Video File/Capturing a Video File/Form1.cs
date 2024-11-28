using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Capturing_a_Video_File
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

            icImagingControl1.LiveStart();
        }

		//<<startlive
        private void btnStartLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStart();
        }
		//>>

		//<<stoplive
        private void btnStopLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStop();
        }
		//>>

        private void btnCaptureVideo_Click(object sender, EventArgs e)
        {
            SaveVideoForm frm = new SaveVideoForm(icImagingControl1);
            frm.ShowDialog();
        }

      
    }
}