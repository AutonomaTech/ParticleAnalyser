using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Capturing_a_Video_File
{
    public partial class SaveVideoForm : Form
    {
        public SaveVideoForm(TIS.Imaging.ICImagingControl ic )
        {
            InitializeComponent();
            _imagingControl = ic;
        }

        private TIS.Imaging.ICImagingControl _imagingControl;
        private TIS.Imaging.BaseSink _oldSink;
        private bool _oldLiveModeSetting;
        private TIS.Imaging.MediaStreamSink _sink;

		//<<FormLoad
        private void SaveVideoForm_Load(object sender, EventArgs e)
        {
            cboMediaStreamContainer.DataSource = TIS.Imaging.MediaStreamContainer.MediaStreamContainers;

            txtFileName.Text = System.IO.Path.ChangeExtension("video.avi", CurrentMediaStreamContainer.PreferredFileExtension);

            btnStopCapture.Enabled = false;

            fillCodecListItems();
        }
		//>>

        private void btnProperties_Click(object sender, EventArgs e)
        {
            CurrentVideoCodec.ShowPropertyPage();
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            SaveFileDialog dlg = new SaveFileDialog();
            dlg.AddExtension = true;

            string ext = CurrentMediaStreamContainer.PreferredFileExtension;
            dlg.DefaultExt = ext;
            dlg.Filter = CurrentMediaStreamContainer.Name
                        + " Video Files (*." + ext + ")|*." + ext + "||";

            if (dlg.ShowDialog() == DialogResult.OK)
            {
                txtFileName.Text = dlg.FileName;
            }
        }

        private TIS.Imaging.MediaStreamContainer CurrentMediaStreamContainer
        {
            get
            {
                return (TIS.Imaging.MediaStreamContainer)cboMediaStreamContainer.SelectedItem;
            }
        }

        private TIS.Imaging.AviCompressor CurrentVideoCodec
        {
            get
            {
                var codec = (TIS.Imaging.AviCompressor)cboVideoCodec.SelectedItem;
                if (codec == null)
                {
                    return null;
                }

                if (CurrentMediaStreamContainer != null && CurrentMediaStreamContainer.IsCodecSupported( codec ) )
                {
                    return codec;
                }
                else
                {
                    return null;
                }
            }
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void btnStopCapture_Click(object sender, EventArgs e)
        {
            _imagingControl.LiveStop();

            chkPause.Checked = false;
            btnStartCapture.Enabled = true;
            btnStopCapture.Enabled = false;
            btnClose.Enabled = true;

            _imagingControl.Sink = _oldSink;

            if (_oldLiveModeSetting)
                _imagingControl.LiveStart();
        }

		//<<StartCapture
        private void btnStartCapture_Click(object sender, EventArgs e)
        {
            _sink = new TIS.Imaging.MediaStreamSink( CurrentMediaStreamContainer, CurrentVideoCodec, txtFileName.Text );
            _sink.SinkModeRunning = !chkPause.Checked;

            _oldLiveModeSetting = _imagingControl.LiveVideoRunning;
            _oldSink = _imagingControl.Sink;

            _imagingControl.LiveStop();

            _imagingControl.Sink = _sink;

            _imagingControl.LiveStart();

            btnStartCapture.Enabled = false;
            btnStopCapture.Enabled = true;
            btnClose.Enabled = false;
        }
		//>>

		//<<chkpauseclick
        private void chkPause_CheckedChanged(object sender, EventArgs e)
        {
            if (_sink != null)
            {
                _sink.SinkModeRunning = !chkPause.Checked;
            }
        }
        //>>

        private void fillCodecListItems()
        {
            var cont = CurrentMediaStreamContainer;
            if (cont != null)
            {
                cboVideoCodec.DataSource = TIS.Imaging.AviCompressor.AviCompressors.Where(p => cont.IsCodecSupported(p)).ToList();
            }
            else
            {
                cboVideoCodec.DataSource = TIS.Imaging.AviCompressor.AviCompressors;
            }
        }

        private void cboMediaStreamContainer_SelectedIndexChanged(object sender, EventArgs e)
        {
            fillCodecListItems();

            btnProperties.Enabled = CurrentVideoCodec == null ? false : CurrentVideoCodec.PropertyPageAvailable;

            txtFileName.Text = System.IO.Path.ChangeExtension(txtFileName.Text, CurrentMediaStreamContainer.PreferredFileExtension);
        }

        private void cboVideoCodec_SelectedIndexChanged(object sender, EventArgs e)
        {
            btnProperties.Enabled = CurrentVideoCodec.PropertyPageAvailable;
        }


    }
}