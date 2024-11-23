using System;
using System.Windows.Forms;
using TIS.Imaging;
using System.Drawing;

namespace Advanced_Image_Processing
{
    public partial class Form1 : Form
    {
        //<<tagrect
        private struct RECT
        {
            public int Left;
            public int Top;
            public int Right;
            public int Bottom;
        }
        //>>

        //<<globals
        private IFrameQueueBuffer _currentlyDisplayedBuffer;
        private RECT _userROI;
        private bool _userROICommited = false;
        private int _threshold = 0;
        private FrameQueueSink _sink;
        //>>

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load( object sender, EventArgs e )
        {
            // setup stuff
            cmdStart.Enabled = false;
            cmdStop.Enabled = false;
            cmdROICommit.Enabled = false;
            cmdSettings.Enabled = false;

            _sink = new TIS.Imaging.FrameQueueSink(( arg ) => NewBufferCallback(arg), TIS.Imaging.MediaSubtypes.Y800, 5);

            icImagingControl1.Sink = _sink;

            // Disable the live display. This allows to display images
            // from the ring buffer in ICImagingControl//s control window.
            icImagingControl1.LiveDisplay = false;

            // Do not add a OverlayBitmap
            icImagingControl1.OverlayBitmapPosition = PathPositions.None;

            // open device:
            icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml");


            UpdateDeviceSettings();
        }

        /// <summary>
        /// cmdStart_Click
        ///
        /// Starts the display.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmdStart_Click( object sender, EventArgs e )
        {
            try
            {
                icImagingControl1.LiveStart();
                cmdStart.Enabled = false;
                cmdStop.Enabled = true;
                //<<cmdStart_Click
                cmdROICommit.Enabled = true;
                //>>cmdStart_Click
                cmdDevice.Enabled = false;
            }
            catch( Exception ex )
            {
                MessageBox.Show( ex.Message );
            }
        }

        /// <summary>
        /// cmdStop_Click
        ///
        /// Stops the display.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void cmdStop_Click( object sender, EventArgs e )
        {
            //<<cmdStop_Click
            try
            {
                cmdStart.Enabled = true;
                cmdStop.Enabled = false;
                cmdDevice.Enabled = true;

                if( _userROICommited )
                {
                    cmdROICommit_Click( sender, e );
                }

                cmdROICommit.Enabled = false;

                icImagingControl1.LiveStop();

                _currentlyDisplayedBuffer = null;
            }
            catch( Exception ex )
            {
                MessageBox.Show( ex.Message );
            }
            //>>cmdStop_Click
        }

        /// <summary>
        /// cmdDevice_Click
        ///
        /// Shows the device settings dialog and initializes some properties.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<cmddevice
        private void cmdDevice_Click( object sender, EventArgs e )
        {
            icImagingControl1.ShowDeviceSettingsDialog();

            UpdateDeviceSettings();
        }

        /// <summary>
        /// UpdateDeviceSettings
        ///
        /// Setup the sink and some runtime variables.
        /// </summary>
        private void UpdateDeviceSettings()
        {
            cmdStart.Enabled = icImagingControl1.DeviceValid;
            cmdSettings.Enabled = icImagingControl1.DeviceValid;

            if( !icImagingControl1.DeviceValid )
            {
                return;
            }

            // Set the size of ICImagingControl to the width and height
            // of the currently selected video format.
            icImagingControl1.Size = icImagingControl1.VideoFormatCurrent.Size;

            _userROI.Top = 0;
            _userROI.Left = 0;
            _userROI.Bottom = icImagingControl1.Width;
            _userROI.Right = icImagingControl1.Height;
        }
        //>>

        /// <summary>
        /// cmdSettings_Click
        ///
        /// Shows the device settings dialog.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<cmdsettings
        private void cmdSettings_Click( object sender, EventArgs e )
        {
            icImagingControl1.ShowPropertyDialog();
        }
        //>>

        /// <summary>
        /// cmdROICommit_Click
        ///
        /// Handles the commit or resets the ROI.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<cmdroicommit
        private void cmdROICommit_Click( object sender, EventArgs e )
        {
            if( !_userROICommited )
            {
                _userROICommited = true;
                cmdROICommit.Text = "Reset ROI";
            }
            else
            {
                _userROICommited = false;
                cmdROICommit.Text = "Set current ROI";
            }
        }
        //>>


        //<<NewBufferCallback
        FrameQueuedResult NewBufferCallback( IFrameQueueBuffer buffer )
        {
            RECT region = NormalizeRect( _userROI, buffer.FrameType.Size );
            if( !_userROICommited )
            {
                ReceiveFrameInContinuousMode( buffer, region );
            }
            else
            {
                ReceiveFrameInCompareMode( buffer, region );
            }
            return FrameQueuedResult.SkipReQueue;
        }
        //>>

        /// <summary>
        /// ReceiveFrameInContinuousMode
        ///
        /// This function is called if the user has not yet committed an ROI.
        /// 
        /// The Region contains the ROI specified by the current mouse position.
        /// The rectangle specified by Region is drawn in the current buffer.
        /// </summary>
        /// <param name="BufferIndex"></param>
        /// <param name="Region"></param>
		//<<ReceiveFrameInContinuousMode
        private void ReceiveFrameInContinuousMode( IFrameQueueBuffer buffer, RECT Region )
        {
            if( _currentlyDisplayedBuffer != null )
            {
                _sink.QueueBuffer(_currentlyDisplayedBuffer);
            }
            _currentlyDisplayedBuffer = buffer;

            DrawRectangleY8( buffer, Region );

            icImagingControl1.DisplayImageBuffer( _currentlyDisplayedBuffer );
        }
        //>>ReceiveFrameInContinuousMode


        /// <summary>
        /// ReceiveFrameInCompareMode
        ///
        /// This function is called when the user has committed an ROI.
        /// Compares the current DisplayBuffer with the recently copied buffer.
        /// If they differ, sets the copied buffer as DisplayBuffer.
        /// </summary>
        /// <param name="BufferIndex"></param>
        /// <param name="Region"></param>
        //<<ReceiveFrameInCompareMode
        private void ReceiveFrameInCompareMode( IFrameQueueBuffer newFrame, RECT Region )
        {
            IFrameQueueBuffer oldBuffer = _currentlyDisplayedBuffer;

            if( oldBuffer == null || CompareRegion(oldBuffer, newFrame, Region, _threshold) )
            {
                if( oldBuffer != null )
                {
                    _sink.QueueBuffer(oldBuffer);
                }

                _currentlyDisplayedBuffer = newFrame;

                DrawRectangleY8(newFrame, Region);

                icImagingControl1.DisplayImageBuffer(newFrame);
            }
            else
            {
                _sink.QueueBuffer(newFrame);
            }
        }
        //>>ReceiveFrameInCompareMode

        // ----------------------------------------------------------------------------
        // Helpers
        // ----------------------------------------------------------------------------

        /// <summary>
        /// NormalizeRect
        ///
        /// Returns a normalized rectangle based on Val.
        /// Normalized means:
        /// (left <= right, top <= bottom, right < MaxX, bottom < MaxY)
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
		//<<normalizerect
        private RECT NormalizeRect( RECT val, Size fmtDim )
        {
            RECT r = val;
            if( r.Top > r.Bottom )
            {
                int Tmp = r.Top;
                r.Top = r.Bottom;
                r.Bottom = Tmp;
            }

            if( r.Left > r.Right )
            {
                int Tmp = r.Left;
                r.Left = r.Right;
                r.Right = Tmp;
            }

            if( r.Top < 0 )
            {
                r.Top = 0;
            }

            if( r.Left < 0 )
            {
                r.Left = 0;
            }

            if( r.Bottom >= fmtDim.Height )
            {
                r.Bottom = fmtDim.Height - 1;
            }

            if( r.Right >= fmtDim.Width )
            {
                r.Right = fmtDim.Width - 1;
            }
            return r;
        }
        //>>

        /// <summary>
        /// CompareRegion
        ///
        /// Compares the contents of Arr with Arr2 in the rectangle
        /// defined by Region. If both arrays differ by more then the
        /// Threshold value,  the function returns true, otherwise false.
        /// </summary>
        /// <param name="Buf"></param>
        /// <param name="Buf2"></param>
        /// <param name="Region"></param>
        /// <param name="Threshold"></param>
        /// <returns></returns>
        //<<compareregion
        private bool CompareRegion( IFrame buf, IFrame buf2, RECT region, int threshold )
        {
            int PixelCount = (region.Bottom - region.Top) * (region.Right - region.Left);
            if (PixelCount <= 0)
            {
                return false;
            }

            long greyscaleDifferenceAccu = 0;
            for( int y = region.Top; y <= region.Bottom; y++ )
            {
                for( int x = region.Left; x <= region.Right; x++ )
                {
                    greyscaleDifferenceAccu += Math.Abs( GetY8PixelAt( buf, x, y ) - GetY8PixelAt( buf2, x, y ) );
                }
            }
            var greyscaleDifference = greyscaleDifferenceAccu / PixelCount;
            if( greyscaleDifference > threshold )
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        //>>

        private static unsafe int GetY8PixelAt( IFrame frm, int x, int y )
        {
            var type = frm.FrameType;
            int bytesPerLine = type.Size.Width * 1;

            var ptr = frm.Ptr;
            return *(ptr + y * bytesPerLine + x);
        }

        private static unsafe void SetY8PixelAt( IFrame frm, int x, int y, byte newValue )
        {
            var type = frm.FrameType;
            int bytesPerLine = type.Size.Width * 1;

            var ptr = frm.Ptr;
            *(ptr + y * bytesPerLine + x) = newValue;
        }


        /// <summary>
        /// IcImagingControl1_MouseDown
        ///
        /// MouseDown event. Resets the user ROI,
        /// if the left mouse button is pressed.
        /// </summary>
        /// <param name="Buf"></param>
        /// <param name="Region"></param>
        private void DrawRectangleY8( IFrameQueueBuffer buf, RECT region )
        {
            const int RECT_COLOR = 255;

            for( int x = region.Left; x <= region.Right; x++ )
            {
                SetY8PixelAt( buf, x, region.Top, RECT_COLOR );
            }

            for( int x = region.Left; x <= region.Right; x++ )
            {
                SetY8PixelAt( buf, x, region.Bottom, RECT_COLOR );
            }

            for( int y = region.Top; y <= region.Bottom; y++ )
            {
                SetY8PixelAt( buf, region.Left, y, RECT_COLOR );
            }

            for( int y = region.Top; y <= region.Bottom; y++ )
            {
                SetY8PixelAt( buf, region.Right, y, RECT_COLOR );
            }
        }

        /// <summary>
        /// IcImagingControl1_MouseDown
        ///
        /// MouseDown event. Resets the user ROI,
        /// if the left mouse button is pressed.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void icImagingControl1_MouseDown( object sender, MouseEventArgs e )
        {
            //<<mousedown
            if( !_userROICommited && (e.Button == MouseButtons.Left) )
            {
                _userROI.Left = e.Location.X;
                _userROI.Top = e.Location.Y;
            }
            //>>
        }

        /// <summary>
        /// IcImagingControl1_MouseMove
        ///
        /// MouseMove event. Sets the user ROI to the current mouse cursor
        /// position, if the left mouse button is pressed.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void icImagingControl1_MouseMove( object sender, MouseEventArgs e )
        {
            //<<mousemove
            if( !_userROICommited && (e.Button == MouseButtons.Left) )
            {
                _userROI.Right = e.Location.X;
                _userROI.Bottom = e.Location.Y;
            }
            //>>	
        }

        /// <summary>
        /// IcImagingControl1_MouseUp
        ///
        /// MouseUp event. Sets the user ROI, if the left
        /// mouse button is released.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void icImagingControl1_MouseUp( object sender, MouseEventArgs e )
        {
            //<<mouseup
            if( !_userROICommited && !(e.Button == MouseButtons.Left) )
            {
                _userROI.Right = e.Location.X;
                _userROI.Bottom = e.Location.Y;
            }
            //>>
        }

        private void sldThresholdSlider_Scroll( object sender, EventArgs e )
        {
            _threshold = sldThresholdSlider.Value;
        }
    }
}