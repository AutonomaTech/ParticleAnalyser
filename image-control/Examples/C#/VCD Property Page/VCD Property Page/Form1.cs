using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace VCD_Property_Page
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            // Check whether a valid video capture device has been selected,
            // otherwise show the device settings dialog
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            icImagingControl1.LiveStart();
        }

        private void cmdSelectDevice_Click(object sender, EventArgs e)
        {
            // The device settings dialog needs the live mode to be stopped
            if (icImagingControl1.LiveVideoRunning)
            {
                icImagingControl1.LiveStop();
            }

            icImagingControl1.ShowDeviceSettingsDialog();

            icImagingControl1.LiveStart();
        }

        private void cmdShowMyDialog_Click(object sender, EventArgs e)
        {
            // Show our VCD Property dialog
            VCDPropertiesDlg vcdPropDlg = new VCDPropertiesDlg( icImagingControl1 );
            vcdPropDlg.ShowDialog();
            vcdPropDlg.Dispose();
        }

        private void cmdShowOriginalDialog_Click(object sender, EventArgs e)
        {
            // Show the builtin property dialog
            icImagingControl1.ShowPropertyDialog();
        }
    }
}