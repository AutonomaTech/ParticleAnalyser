using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Capturing_an_AVI_File
{
    public partial class WriteAvi : Form
    {
        public WriteAvi(TIS.Imaging.ICImagingControl ic )
        {
            InitializeComponent();
            _imagingControl = ic;
        }

        private TIS.Imaging.ICImagingControl _imagingControl;
        private TIS.Imaging.BaseSink _savedSink;
        private bool _wasRunning = false;

        /// <summary>
        /// writeavi_Load
        ///
        /// Display all available video codecs in a combo box
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<Formload
        private void WriteAvi_Load(object sender, EventArgs e)
        {
            cboVideoCodec.DataSource = TIS.Imaging.AviCompressor.AviCompressors;

            // Show the first codec in the combobox.
            cboVideoCodec.SelectedIndex = 0;
            cmdStartCapture.Enabled = true;
            cmdStopCapture.Enabled = false;
        }
        //>>

        /// <summary>
        /// cboVideoCodec_SelectedValueChanged
        ///
        /// Handle the change of the current selection in the cvbVideoCodec combo box. If
        /// the selection has changed, it is checked whether the codec as a properties
        /// dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<cboVideoCodec_Click
        private void cboVideoCodec_SelectedIndexChanged(object sender, EventArgs e)
        {
            TIS.Imaging.AviCompressor Codec;
            // Retrieve the codec from the cboVideoCodec combobox.
            Codec = (TIS.Imaging.AviCompressor)cboVideoCodec.SelectedItem;

            //Check for the configuration dialog.
            if (Codec.PropertyPageAvailable)
            {
                cmdShowPropertyPage.Enabled = true;
            }
            else
            {
                cmdShowPropertyPage.Enabled = false;
            }
        }
		//>>

        /// <summary>
        /// cmdShowPropertyPage_Click
        ///
        /// Show the property dialog of the currently selected codec.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<cmdShowPropertyPage_Click
        private void cmdShowPropertyPage_Click(object sender, EventArgs e)
        {
            TIS.Imaging.AviCompressor Codec;
            // Retrieve the codec from the cboVideoCodec combobox.
            Codec = (TIS.Imaging.AviCompressor)cboVideoCodec.SelectedItem;
            Codec.ShowPropertyPage();
        }
		//>>

        /// <summary>
        /// cmdFilename_Click
        ///
        /// Select a filename for the AVI file.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmdFilename_Click(object sender, EventArgs e)
        {
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();
            saveFileDialog1.Filter = "avi files (*.avi)|*.avi|All files (*.*)|*.*";
            saveFileDialog1.FilterIndex = 1;
            saveFileDialog1.RestoreDirectory = true;

            if (saveFileDialog1.ShowDialog() == DialogResult.OK)
            {
                txtFilename.Text = saveFileDialog1.FileName;
            }
        }

        /// <summary>
        /// cmdStartCapture
        ///
        /// Start avi capture with the selected filename and codec.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<StartCapture
        private void cmdStartCapture_Click(object sender, EventArgs e)
        {
            if (txtFilename.Text == "")
            {
                MessageBox.Show("Please select an AVI filename first.");
                return;
            }
            _wasRunning = _imagingControl.LiveVideoRunning;

            if( _imagingControl.LiveVideoRunning )
                _imagingControl.LiveStop();

            _savedSink = _imagingControl.Sink;
            _imagingControl.Sink = new TIS.Imaging.MediaStreamSink( (TIS.Imaging.AviCompressor)cboVideoCodec.SelectedItem, txtFilename.Text );
            try
            {
                _imagingControl.LiveStart();
                cmdStopCapture.Enabled = true;
                cmdStartCapture.Enabled = false;
            }
            catch
            {
                _imagingControl.Sink = _savedSink;
                if( _wasRunning )
                {
                    _imagingControl.LiveStart();
                }
            }
        }
		//>>

        /// <summary>
        /// cmdStopCapture_Click
        ///
        /// Stop video capture.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmdStopCapture_Click(object sender, EventArgs e)
        {
            _imagingControl.LiveStop();
            _imagingControl.Sink = _savedSink;
            cmdStopCapture.Enabled = false;
            cmdStartCapture.Enabled = true;

            if( _wasRunning )
                _imagingControl.LiveStart();
        }

        /// <summary>
        /// chkPause_Click
        ///
        /// Pause or restart the avi capture according to the value in chkPause.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<chkpauseclick
        private void chkPause_CheckedChanged(object sender, EventArgs e)
        {
            _imagingControl.Sink.SinkModeRunning = chkPause.Checked;
        }
		//>>

    }
}