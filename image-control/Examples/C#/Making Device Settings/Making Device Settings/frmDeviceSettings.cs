using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace MakingDeviceSettings
{
	public partial class frmDeviceSettings : Form
	{
        //<<globals
        private TIS.Imaging.ICImagingControl imagingControl;
        private string deviceState;
        private const string not_available_text = "n\\a";
		//>>

		public frmDeviceSettings( TIS.Imaging.ICImagingControl ic )
		{
            imagingControl = ic;

            InitializeComponent();
		}

		// ------------------------------------------------------------------------------
		// Form events
		// ------------------------------------------------------------------------------

		//
		// Form_Load
		//
		// Fill the Video Sources combo box with names of all available
		// video capture devices and select the first one. This will trigger
		// a click event on the Video Sources combo box and open the device.
		//
		//<<formload
		private void frmDeviceSettings_Load( object sender, EventArgs e )
		{
			if( imagingControl.DeviceValid )
			{
				if( imagingControl.LiveVideoRunning )
				{
					lblErrorMessage.Text = "The device settings dialog is not available while the live video is running.\n\nStop the live video first.";
					lblErrorMessage.AutoSize = false;
					lblErrorMessage.Padding = new Padding( 8 );
					lblErrorMessage.SetBounds( 0, 0, 100, cmdOK.Top );
					lblErrorMessage.Dock = DockStyle.Top;
					lblErrorMessage.Visible = true;
					return;
				}
				else
				{
					lblErrorMessage.Visible = false;
				}
			}

			SaveDeviceSettings();

			UpdateDevices();
		}
		//>>

		private void SaveDeviceSettings()
		{
			deviceState = imagingControl.DeviceState;
		}

		private void RestoreDeviceSettings()
		{
			try
			{
				imagingControl.DeviceState = deviceState;
			}
			catch (System.Exception)
			{
			}			
		}

		// ------------------------------------------------------------------------------
		// UI Update
		// ------------------------------------------------------------------------------

		//
		// UpdateDevices
		//
		// Fills cboDevice
		//
		//<<updateDevice
		private void UpdateDevices()
		{
			cboDevice.Items.Clear();
			if( imagingControl.Devices.Length > 0 )
			{
				foreach( object Item in imagingControl.Devices )
				{
					cboDevice.Items.Add( Item.ToString() );
				}

				if( imagingControl.DeviceValid )
				{
					cboDevice.SelectedItem = imagingControl.Device;
				}
				else
				{
					cboDevice.SelectedIndex = 0;
				}
				cboDevice.Enabled = true;
			}
			else
			{
				cboDevice.Items.Add( not_available_text );
				cboDevice.Enabled = false;
				cboDevice.SelectedIndex = 0;
			}
		}
		//>>

		//
		// UpdateVideoNorms
		//
		// Fills cboVideoNorm
		//
		//<<updateVideoNorm
		private void UpdateVideoNorms()
		{
			cboVideoNorm.Items.Clear();
			if( imagingControl.VideoNormAvailable )
			{
				foreach( object Item in imagingControl.VideoNorms )
				{
					cboVideoNorm.Items.Add( Item.ToString() );
				}

				cboVideoNorm.SelectedItem = imagingControl.VideoNorm.ToString();
				cboVideoNorm.Enabled = true;
			}
			else
			{
				cboVideoNorm.Items.Add( not_available_text );
				cboVideoNorm.Enabled = false;
				cboVideoNorm.SelectedIndex = 0;
			}
		}
		//>>

		//
		// UpdateVideoFormats
		//
		// Fills cboVideoFormat
		//
		//<<updateVideoFormat
		private void UpdateVideoFormats()
		{
			cboVideoFormat.Items.Clear();
			if( imagingControl.DeviceValid )
			{
				foreach( object Item in imagingControl.VideoFormats )
				{
					cboVideoFormat.Items.Add( Item.ToString() );
				}

				cboVideoFormat.SelectedItem = imagingControl.VideoFormat.ToString();
				cboVideoFormat.Enabled = true;
			}
			else
			{
				cboVideoFormat.Items.Add( not_available_text );
				cboVideoFormat.Enabled = false;
				cboVideoFormat.SelectedIndex = 0;
			}
		}
		//>>

		//
		// UpdateInputChannels
		//
		// Fills cboInputChannel
		//
		//<<updateInputChannel
		private void UpdateInputChannels()
		{
			cboInputChannel.Items.Clear();
			if( imagingControl.InputChannelAvailable )
			{
				foreach( object Item in imagingControl.InputChannels )
				{
					cboInputChannel.Items.Add( Item.ToString() );
				}

				cboInputChannel.SelectedItem = imagingControl.InputChannel;
				cboInputChannel.Enabled = true;
			}
			else
			{
				cboInputChannel.Items.Add( not_available_text );
				cboInputChannel.Enabled = false;
				cboInputChannel.SelectedIndex = 0;
			}
		}
		//>>

		//
		// UpdateFrameRates
		//
		// Fills cboFrameRates
		//
		//<<updateFrameRates
		private void UpdateFrameRates()
		{
			cboFrameRate.Items.Clear();
			if( imagingControl.DeviceFrameRateAvailable )
			{
				foreach( object Item in imagingControl.DeviceFrameRates )
				{
					cboFrameRate.Items.Add( Item.ToString() );
				}

				cboFrameRate.SelectedItem = ( imagingControl.DeviceFrameRate ).ToString();
				cboFrameRate.Enabled = true;
			}
			else
			{
				cboFrameRate.Items.Add( not_available_text );
				cboFrameRate.Enabled = false;
				cboFrameRate.SelectedIndex = 0;
			}
		}
		//>>

		//
		// UpdateFlip
		//
		// updates the flip checkboxes
		//
		//<<updateFlip
		private void UpdateFlip()
		{
			if( imagingControl.DeviceFlipHorizontalAvailable )
			{
				chkFlipH.Enabled = true;
				if( imagingControl.DeviceFlipHorizontal )
				{
					chkFlipH.Checked = true;
				}
				else
				{
					chkFlipH.Checked = false;
				}
			}
			else
			{
				chkFlipH.Enabled = false;
				chkFlipH.Checked = false;
			}

			if( imagingControl.DeviceFlipVerticalAvailable )
			{
				chkFlipV.Enabled = true;
				if( imagingControl.DeviceFlipVertical )
				{
					chkFlipV.Checked = true;
				}
				else
				{
					chkFlipV.Checked = false;
				}
			}
			else
			{
				chkFlipV.Enabled = false;
				chkFlipV.Checked = false;
			}
		}
		//>


		// ------------------------------------------------------------------------------
		// UI events
		// ------------------------------------------------------------------------------

		//
		// cboDevice_SelectedIndexChanged
		//
		// Get available inputs and video formats for the selected
		// device and enter the information in the respective combo
		// boxes.
		//
		//<<cboDevice
		private void cboDevice_SelectedIndexChanged( object sender, System.EventArgs e )
		{
			try
			{
				// Open the device
				if( cboDevice.Enabled )
				{
					imagingControl.Device = cboDevice.Text;

                    string serial;
                    if( imagingControl.DeviceCurrent.GetSerialNumber(out serial) )
                    {
                        txtSerial.Text = serial;
                    }
                    else
                    {
                        txtSerial.Text = not_available_text;
                    }
                }
                // Get supported video norms, formats and inputs
                UpdateVideoNorms();
				UpdateInputChannels();
				UpdateFlip();
			}
			catch( Exception ex )
			{
				MessageBox.Show( ex.Message );
			}
		}
		//>>

		//
		// cboVideoNorm_SelectedIndexChanged
		//
		// Select a video norm.
		//
		//<<cboVideoNorm
		private void cboVideoNorm_SelectedIndexChanged( object sender, System.EventArgs e )
		{
			try
			{
				if( cboVideoNorm.Enabled )
				{
					imagingControl.VideoNorm = cboVideoNorm.Text;
				}

				UpdateVideoFormats();
			}
			catch( Exception ex )
			{
				MessageBox.Show( ex.Message );
			}
		}
		//>>

		//
		// cboInputChannel_SelectedIndexChanged
		//
		// Select an input channel.
		//
		//<<cboInputChannel
		private void cboInputChannel_SelectedIndexChanged( object sender, System.EventArgs e )
		{
			try
			{
				if( cboInputChannel.Enabled )
				{
					imagingControl.InputChannel = cboInputChannel.Text;
				}
			}
			catch( Exception ex )
			{
				MessageBox.Show( ex.Message );
			}
		}
		//>>

		//
		// cboVideoFormat_SelectedIndexChanged
		//
		// Select a video format.
		//	
		//<<cboVideoFormat
		private void cboVideoFormat_SelectedIndexChanged( object sender, System.EventArgs e )
		{
			try
			{
				if( cboVideoFormat.Enabled )
				{
					imagingControl.VideoFormat = cboVideoFormat.Text;
				}

				UpdateFrameRates();
			}
			catch( Exception ex )
			{
				MessageBox.Show( ex.Message );
			}
		}
		//>>

		//
		// cboFrameRate_SelectedIndexChanged
		//
		// Select a frame rate
		//
		//<<cboFrameRate
		private void cboFrameRate_SelectedIndexChanged( object sender, System.EventArgs e )
		{
			try
			{
				if( cboFrameRate.Enabled )
				{
					imagingControl.DeviceFrameRate = (float)( System.Convert.ToDouble( cboFrameRate.Text ) );
				}
			}
			catch( Exception ex )
			{
				MessageBox.Show( ex.Message );
			}
		}
		//>>

		//
		// chkFlipV
		//
		// Switch flip vertical on/off
		//
		//<<chkFlipV
		private void chkFlipV_CheckedChanged( object sender, System.EventArgs e )
		{
			if( imagingControl.DeviceFlipVerticalAvailable )
			{
				imagingControl.DeviceFlipVertical = ( chkFlipV.Checked == true );
			}
		}
		//>>

		//
		// chkFlipH
		//
		// Switch flip horizontal on/off
		//
		private void chkFlipH_CheckedChanged( object sender, System.EventArgs e )
		{
			if( imagingControl.DeviceFlipHorizontalAvailable )
			{
				imagingControl.DeviceFlipHorizontal = ( chkFlipH.Checked == true );
			}
		}


		// ------------------------------------------------------------------------------
		// Buttons
		// ------------------------------------------------------------------------------

		//
		// cmdOK_Click
		//
		// Close form.
		//
		//<<cmdOK
		private void cmdOK_Click( object sender, System.EventArgs e )
		{
			this.Close();
		}
		//>>

		//
		// cmdCancel_Click
		//
		// Close form and set canceled to true.
		//
		//<<cmdCancel
		private void cmdCancel_Click( object sender, System.EventArgs e )
		{
			RestoreDeviceSettings();

			this.Close();
		}
		//>>
	}
}