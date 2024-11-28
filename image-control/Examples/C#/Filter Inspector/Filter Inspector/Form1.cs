using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Filter_Inspector
{
    public partial class Form1 : Form
    {
        private System.Collections.Specialized.StringCollection modulePathCollection;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
//<<formLoad
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            // Disable all overlays, so that they do not influence the
            // color format of the image stream.
            icImagingControl1.OverlayBitmapPosition = TIS.Imaging.PathPositions.None;

            // Start live mode.
            icImagingControl1.LiveStart();
//>>

//<<enumModules
            // Use a collection to save the full paths to the filter modules.
            modulePathCollection = new System.Collections.Specialized.StringCollection();

            // For each filter info:
            // - Check if the filter's path is already in the module paths collection
            // - If not, add the module name to the filter module list box.
            foreach (TIS.Imaging.FrameFilterInfo ffi in TIS.Imaging.FrameFilterInfo.FrameFilterInfos)
            {
                if (modulePathCollection.IndexOf(ffi.ModulePath) < 0)
                {
                    lstFrameFilterModules.Items.Add(ffi.ModuleName);
                    modulePathCollection.Add(ffi.ModulePath);
                }
            }
//>>
        }

        private void btnDevice_Click(object sender, EventArgs e)
        {
            // If live mode is active, stop.
            bool wasLive = icImagingControl1.LiveVideoRunning;
            if (wasLive)
            {
                icImagingControl1.LiveStop();
            }

            // Show device selection dialog.
            icImagingControl1.ShowDeviceSettingsDialog();

            // If live mode was active, restart.
            if (wasLive)
            {
                icImagingControl1.LiveStart();
            }		
        }

        private void btnProperties_Click(object sender, EventArgs e)
        {
            icImagingControl1.ShowPropertyDialog();
        }

        private void btnDialog_Click(object sender, EventArgs e)
        {
            icImagingControl1.DeviceFrameFilters[0].ShowDialog();
        }

        private void btnRemove_Click(object sender, EventArgs e)
        {
            // If live mode is active, stop.
            bool wasLive = icImagingControl1.LiveVideoRunning;
            if (wasLive)
            {
                icImagingControl1.LiveStop();
            }

            // Show device selection dialog.
            icImagingControl1.DeviceFrameFilters.Clear();

            // If live mode was active, restart.
            if (wasLive)
            {
                icImagingControl1.LiveStart();
            }

            lblSelectedFilter.Text = "<no filter>";
            btnDialog.Enabled = false;
            btnRemove.Enabled = false;
            lstFrameFilters.SelectedItem = null;
        }

        private void btnStartLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStart();
        }

        private void btnStopLive_Click(object sender, EventArgs e)
        {
            icImagingControl1.LiveStop();
        }

//<<enumFilters
        private void lstFrameFilterModules_SelectedIndexChanged(object sender, EventArgs e)
        {
            // Get the full path to the selected module from the ModulePaths collection.
            string selectedModulePath = modulePathCollection[lstFrameFilterModules.SelectedIndex];

            lstFrameFilters.Items.Clear();
            foreach (TIS.Imaging.FrameFilterInfo ffi in TIS.Imaging.FrameFilterInfo.FrameFilterInfos)
            {
                if (ffi.ModulePath == selectedModulePath)
                {
                    lstFrameFilters.Items.Add(ffi);
                }
            }
        }
//>>

        private void lstFrameFilters_SelectedIndexChanged(object sender, EventArgs e)
        {
//<<createInstance
            // Get the selected FrameFilterInfo object
            TIS.Imaging.FrameFilterInfo ffi = (TIS.Imaging.FrameFilterInfo)lstFrameFilters.SelectedItem;

            if (ffi != null)
            {
                // Create the new FrameFilter instance.
                TIS.Imaging.FrameFilter newFrameFilter = TIS.Imaging.FrameFilter.Create(ffi);
//>>
//<<setFilter
                // If live mode is active, stop.
                bool wasLive = icImagingControl1.LiveVideoRunning;
                if (wasLive)
                {
                    icImagingControl1.LiveStop();
                }

                // Set the new frame filter.
                icImagingControl1.DeviceFrameFilters.Clear();
                icImagingControl1.DeviceFrameFilters.Add(newFrameFilter);

                // If live mode was active, restart.
                if (wasLive)
                {
                    icImagingControl1.LiveStart();
                }
//>>
                lblSelectedFilter.Text = newFrameFilter.Name;
                btnDialog.Enabled = newFrameFilter.HasDialog;
                btnRemove.Enabled = true;
            }
        }


    }
}