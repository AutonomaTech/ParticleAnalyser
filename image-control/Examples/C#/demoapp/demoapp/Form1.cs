using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

using TIS.Imaging;

namespace demoapp
{
    public partial class Form1 : Form
    {
        private FrameSnapSink _snapSink;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load( object sender, EventArgs e )
        {
            LoadLastUsedDevice();
            UpdateControls();

            _snapSink = new FrameSnapSink(); ;

            icImagingControl1.Sink = _snapSink;
        }

        /// <summary>
        /// exitToolStripMenuItem_Click
        ///
        /// Exit the demo application
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void exitToolStripMenuItem_Click( object sender, EventArgs e )
        {
            Close();
        }

        /// <summary>
        /// deviceToolStripMenuItem_Click
        ///
        /// Show the device selection dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void deviceToolStripMenuItem_Click( object sender, EventArgs e )
        {
            SelectDevice();
        }

        /// <summary>
        /// playToolStripMenuItem_Click
        ///
        /// Handler for the play menu item. StartLiveVideo() sub is called.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void playToolStripMenuItem_Click( object sender, EventArgs e )
        {
            StartLiveVideo();
        }

        /// <summary>
        /// stopToolStripMenuItem_Click
        ///
        /// Stop the live video display by a call to StopLiveVideo() sub.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void stopToolStripMenuItem_Click( object sender, EventArgs e )
        {
            StopLiveVideo();
        }

        private void Form1_Resize( object sender, EventArgs e )
        {
            icImagingControl1.Size = ClientSize;
        }

        /// <summary>
        /// imageToolStripMenuItem_Click
        ///
        /// Show the video and camera properties dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void imageToolStripMenuItem_Click( object sender, EventArgs e )
        {
            ShowProperties();
        }

        /// <summary>
        /// writeAviToolStripMenuItem_Click
        ///
        /// Show the AVI capture dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void writeAviToolStripMenuItem_Click( object sender, EventArgs e )
        {
            ShowAVIDialog();
        }

        /// <summary>
        /// saveImageToolStripMenuItem_Click
        ///
        /// Handler for the SaveImage menu. SaveImage() Sub is called.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void saveImageToolStripMenuItem_Click( object sender, EventArgs e )
        {
            SaveImage();
        }

        private void externalTriggerToolStripMenuItem_Click( object sender, EventArgs e )
        {
            ToggleTrigger();
        }

        /// <summary>
        /// UpdateControls
        ///
        /// UpdateControls enables and disables the menuitems of the application depending of
        /// camera is connected to the computer or not.
        /// </summary>
        private void UpdateControls()
        {
            bool IsDeviceValid = false;

            IsDeviceValid = icImagingControl1.DeviceValid;
            imageToolStripMenuItem.Enabled = IsDeviceValid;
            writeAviToolStripMenuItem.Enabled = IsDeviceValid;
            saveImageToolStripMenuItem.Enabled = IsDeviceValid;
            playToolStripMenuItem.Enabled = IsDeviceValid;
            stopToolStripMenuItem.Enabled = IsDeviceValid;
            ToolbarPlayButton.Enabled = IsDeviceValid;
            ToolBarStopButton.Enabled = IsDeviceValid;

            externalTriggerToolStripMenuItem.Enabled = false;
            ToolBarTriggerButton.Enabled = false;


            if( IsDeviceValid )
            {
                playToolStripMenuItem.Enabled = !icImagingControl1.LiveVideoRunning;
                stopToolStripMenuItem.Enabled = !playToolStripMenuItem.Enabled;
                ToolbarPlayButton.Enabled = !icImagingControl1.LiveVideoRunning;
                ToolBarStopButton.Enabled = !ToolbarPlayButton.Enabled;

                if( icImagingControl1.DeviceTriggerAvailable )
                {
                    externalTriggerToolStripMenuItem.Enabled = true;
                    ToolBarTriggerButton.Enabled = true;
                    externalTriggerToolStripMenuItem.Checked = icImagingControl1.DeviceTrigger;
                    ToolBarTriggerButton.Checked = icImagingControl1.DeviceTrigger;
                }

                if( icImagingControl1.InputChannelAvailable )
                {
                    //Create the sub menus that allow to select the input channels.
                    ToolBarInputChannel.DropDownItems.Clear();
                    inputChannelsToolStripMenuItem.DropDown.Items.Clear();
                    inputChannelsToolStripMenuItem.Enabled = true;
                    ToolBarInputChannel.Enabled = true;

                    foreach( InputChannel s in icImagingControl1.InputChannels )
                    {
                        System.Windows.Forms.ToolStripItem mitem = null;

                        // Add the input channel as menu item to the main menu
                        //mitem = inputChannelsToolStripMenuItem.DropDown.Items.Add(s.ToString());

                        mitem = inputChannelsToolStripMenuItem.DropDownItems.Add( s.ToString() );

                        if( icImagingControl1.InputChannel == s.ToString() )
                        {
                            ToolStripMenuItem i = (ToolStripMenuItem)mitem;
                            i.Checked = true;
                        }
                        mitem.Click += new System.EventHandler( mnuInputChannelChild_Click );

                        // Add the input channel as menu item to the toolbar button's context menu
                        mitem = ToolBarInputChannel.DropDown.Items.Add( s.ToString() );
                        if( icImagingControl1.InputChannel == s.ToString() )
                        {
                            ToolStripMenuItem i = (ToolStripMenuItem)mitem;
                            i.Checked = true;
                        }
                        mitem.Click += new System.EventHandler( mnuInputChannelChild_Click );
                    }
                }
                else
                {
                    // Remove the input channels from the submenus.
                    ToolBarInputChannel.DropDownItems.Clear();
                    inputChannelsToolStripMenuItem.DropDown.Items.Clear();
                    inputChannelsToolStripMenuItem.Enabled = false;
                    ToolBarInputChannel.Enabled = false;
                }

            }
            ToolBarSnapButton.Enabled = IsDeviceValid;
            ToolBarAVIButton.Enabled = IsDeviceValid;
            ToolBarPropertiesButton.Enabled = IsDeviceValid;
        }

        /// <summary>
        /// toolStrip1_ItemClicked
        ///
        /// Handler for the toolbar play button. StartLiveVideo sub is called.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void toolStrip1_ItemClicked( object sender, ToolStripItemClickedEventArgs e )
        {
            // Evaluate the Button property to determine which button was clicked.
            switch( toolStrip1.Items.IndexOf( e.ClickedItem ) )
            {
            case 0:
                StartLiveVideo();
                break;
            case 1:
                StopLiveVideo();
                break;
            case 3:
                SaveImage();
                break;
            case 4:
                ShowAVIDialog();
                break;
            case 6:
                SelectDevice();
                break;
            case 7:
                ShowProperties();
                break;
            case 10:
                ToggleTrigger();
                break;
            }
        }

        /// <summary>
        /// StartLiveVideo
        ///
        /// Start the live video display and change the button states of the
        /// play and stop button.
        /// </summary>
        private void StartLiveVideo()
        {
            if( icImagingControl1.DeviceValid )
            {
                icImagingControl1.LiveStart();
                playToolStripMenuItem.Enabled = false;
                stopToolStripMenuItem.Enabled = true;
                ToolbarPlayButton.Enabled = false;
                ToolBarStopButton.Enabled = true;
                ToolBarSnapButton.Enabled = true;
            }
        }

        /// <summary>
        /// StopLiveVideo
        ///
        /// Stop the live video display and change the button states of the
        /// play and stop button.
        /// </summary>
        private void StopLiveVideo()
        {
            if( icImagingControl1.DeviceValid )
            {
                icImagingControl1.LiveStop();
                playToolStripMenuItem.Enabled = true;
                stopToolStripMenuItem.Enabled = false;
                ToolbarPlayButton.Enabled = true;
                ToolBarStopButton.Enabled = false;
                ToolBarSnapButton.Enabled = false;
            }
        }

        /// <summary>
        /// SelectDevice
        ///
        /// Show the device selection dialog.
        /// </summary>
        private void SelectDevice()
        {
            if (icImagingControl1.LiveVideoRunning)
            {
                icImagingControl1.LiveStop();
                icImagingControl1.ShowDeviceSettingsDialog();
                if (icImagingControl1.DeviceValid)
                {
                    icImagingControl1.LiveStart();
                }
            }
            else
            {
                icImagingControl1.ShowDeviceSettingsDialog();
            }

            UpdateControls();
            SetWindowSizeToImagesize();

            DeviceSettings.SaveSelectedDevice( icImagingControl1 );

        }

        /// <summary>
        /// ShowProperties
        ///
        /// Show the property dialog of the current video capture device.
        /// </summary>
        private void ShowProperties()
        {
            if( icImagingControl1.DeviceValid )
            {
                icImagingControl1.ShowPropertyDialog();
                UpdateControls();
            }
        }

        /// <summary>
        /// SaveImage
        ///
        /// Snap (capture) an image from the video stream and save it to disc.
        /// </summary>
        private void SaveImage()
        {
            if (!icImagingControl1.LiveVideoRunning)
            {
                return;
            }

            IFrameQueueBuffer buf;
            try
            {
                // Snap (capture) an image to the memory
                buf = _snapSink.SnapSingle(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex)
            {
                MessageBox.Show("Failed to snap image due to: " + ex);
                return;
            }


            // Call the save file dialog to enter the file name of the image
            SaveFileDialog saveFileDialog1 = new SaveFileDialog();
            saveFileDialog1.Filter = "bmp files (*.bmp)|*.bmp|All files (*.*)|*.*";
            saveFileDialog1.FilterIndex = 1;
            saveFileDialog1.RestoreDirectory = true;

            if( saveFileDialog1.ShowDialog() == DialogResult.OK )
            {
                // Save the image.
                buf.SaveAsBitmap( saveFileDialog1.FileName );
            }
        }

        /// <summary>
        /// ShowAVIDialog
        ///
        /// Show the dialog for AVI capture
        /// </summary>
        private void ShowAVIDialog()
        {
            if( icImagingControl1.DeviceValid )
            {
                AviDialog AVIDlg = new AviDialog( icImagingControl1 );
                AVIDlg.ShowDialog();
                AVIDlg.Dispose();
                AVIDlg = null;
            }
        }

        /// <summary>
        /// ToggleTrigger
        ///
        /// Enable or disable the external trigger.
        /// </summary>
        private void ToggleTrigger()
        {
            if( icImagingControl1.DeviceValid )
            {
                if( icImagingControl1.DeviceTriggerAvailable )
                {
                    bool trigger = !icImagingControl1.DeviceTrigger;

                    if( icImagingControl1.LiveVideoRunning )
                    {
                        icImagingControl1.LiveStop();
                        icImagingControl1.DeviceTrigger = trigger;
                        icImagingControl1.LiveStart();
                    }
                    else
                    {
                        icImagingControl1.DeviceTrigger = trigger;
                    }

                    externalTriggerToolStripMenuItem.Checked = trigger;
                    ToolBarTriggerButton.Checked = trigger;
                }
            }
        }

        /// <summary>
        /// SetWindowSizeToImagesize
        ///
        /// Resize the application to the size of the video format.
        /// </summary>
        private void SetWindowSizeToImagesize()
        {
            if( icImagingControl1.DeviceValid )
            {
                // Change the witdth to a minimum size of 230 pixel. If the width is
                // less than 230 pixel, the toolbar can not be seen completely.
                Width = Math.Max( icImagingControl1.LiveDisplayOutputSize.Width + Width - ClientSize.Width, 230 );
                Height = icImagingControl1.LiveDisplayOutputSize.Height + Height - ClientSize.Height + toolStrip1.Size.Height;
            }
        }

        /// <summary>
        /// mnuInputChannelChild_Click
        ///
        /// If the video capture device supports input channels, submenus
        /// for the input channel selection are created. This is the handler
        /// sub for the input channel submenus.
        /// The menu text if the clicked submenu item is retrieved and assigned
        /// to the InputChannel property of IC15Control.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void mnuInputChannelChild_Click( object sender, System.EventArgs e )
        {
            System.Windows.Forms.ToolStripMenuItem mitem = null;
            mitem = ((System.Windows.Forms.ToolStripMenuItem)(sender));

            foreach( System.Windows.Forms.ToolStripMenuItem Item in inputChannelsToolStripMenuItem.DropDownItems )
            {
                if( Item.Text == mitem.Text )
                    Item.Checked = true;
                else
                    Item.Checked = false;
            }

            foreach( System.Windows.Forms.ToolStripMenuItem Item in ToolBarInputChannel.DropDown.Items )
            {
                if( Item.Text == mitem.Text )
                    Item.Checked = true;
                else
                    Item.Checked = false;
            }


            icImagingControl1.InputChannel = mitem.Text;
            // Save the new input channel to the registry.
            DeviceSettings.SaveSelectedDevice( icImagingControl1 );

        }

        /// <summary>
        /// LoadLastUsedDevice
        ///
        /// Retrieve the last used video capture device from the registry and
        /// open it. The implementation of these functionallity is in the
        /// modul devicesettings.vb
        /// </summary>
        private void LoadLastUsedDevice()
        {
            DeviceSettings.OpenSelectedDevice( icImagingControl1 );
            SetWindowSizeToImagesize();
        }

    }
}