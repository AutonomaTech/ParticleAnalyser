using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Pixelformat
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

			cmdY16.Enabled = icImagingControl1.VideoFormat.StartsWith("Y16");

            icImagingControl1.OverlayBitmapPosition = TIS.Imaging.PathPositions.None;
            icImagingControl1.LiveStart();
        }

        //<<GrabImage
        private TIS.Imaging.IFrameQueueBuffer GrabImage(Guid colorFormat)
        {
            bool wasLive = icImagingControl1.LiveVideoRunning;
            icImagingControl1.LiveStop();

            TIS.Imaging.BaseSink oldSink = icImagingControl1.Sink;

            TIS.Imaging.FrameSnapSink sink = new TIS.Imaging.FrameSnapSink( new TIS.Imaging.FrameType( colorFormat ) );

            icImagingControl1.Sink = sink;

            try
            {
                icImagingControl1.LiveStart();
            }
            catch (TIS.Imaging.ICException ex)
            {
                MessageBox.Show(ex.Message);
                icImagingControl1.Sink = oldSink;
                return null;
            }

            TIS.Imaging.IFrameQueueBuffer rval = null;

            try
            {
                rval = sink.SnapSingle( TimeSpan.FromSeconds( 1 ) );
            }
            catch (TIS.Imaging.ICException ex)
            {
                MessageBox.Show(ex.Message);
            }

            icImagingControl1.LiveStop();

            icImagingControl1.Sink = oldSink;

            if (wasLive)
                icImagingControl1.LiveStart();

            return rval;
        }
        //>>GrabImage

        //<<BufferAccessHelper
        class BufferAccessHelper
        {
            TIS.Imaging.IFrameQueueBuffer buf;

            public BufferAccessHelper( TIS.Imaging.IFrameQueueBuffer buf )
            {
                this.buf = buf;
            }

            public unsafe byte this[int x,int y]
            {
                get
                {
                    byte* ptr = buf.Ptr + y * buf.FrameType.BytesPerLine + x;
                    return *ptr;
                }
                set
                {
                    byte* ptr = buf.Ptr + y * buf.FrameType.BytesPerLine + x;
                    *ptr = value;
                }
            }
        }
        //>>


        private void cmdY800_Click(object sender, EventArgs e)
        {
            TIS.Imaging.IFrameQueueBuffer frame = GrabImage(TIS.Imaging.MediaSubtypes.Y800);
            if ( frame == null) return;

            BufferAccessHelper buf = new BufferAccessHelper( frame );

//<<y800print
            // Y800 is top-down, the first line has index 0
            int y = 0;

            txtOutput.Text = "Image buffer pixel format is Y800\r\n";
            txtOutput.Text += "Pixel 1: " + buf[0, y] + "\r\n";
            txtOutput.Text += "Pixel 2: " + buf[1, y];
//>>

//<<y800edit
            // Set the first pixel to 0 (black)
            buf[0, y] = 0;
            // Set the second pixel to 128 (gray)
            buf[1, y] = 128;
            // Set the third pixel to 255 (white)
            buf[2, y] = 255;

            TIS.Imaging.FrameExtensions.SaveAsBitmap(frame, "Y800.bmp");
//>>
        }

        private void cmdRGB24_Click(object sender, EventArgs e)
        {
            TIS.Imaging.IFrameQueueBuffer frame = GrabImage(TIS.Imaging.MediaSubtypes.RGB24);
            if ( frame == null) return;

            BufferAccessHelper buf = new BufferAccessHelper(frame);

//<<rgb24print
            // RGB24 is bottom-up, the first line has index lines-1
            int y = frame.FrameType.Height - 1;

            txtOutput.Text = "Image buffer pixel format is RGB24\r\n";
            txtOutput.Text += "Pixel 1: ";
            txtOutput.Text += "R=" + buf[0 * 3 + 2, y] + ", ";
            txtOutput.Text += "G=" + buf[0 * 3 + 1, y] + ", ";
            txtOutput.Text += "B=" + buf[0 * 3 + 0, y] + "\r\n";
            txtOutput.Text += "Pixel 2: ";
            txtOutput.Text += "R=" + buf[1 * 3 + 2, y] + ", ";
            txtOutput.Text += "G=" + buf[1 * 3 + 1, y] + ", ";
            txtOutput.Text += "B=" + buf[1 * 3 + 0, y];
//>>

//<<rgb24edit
            // Set the first pixel to red (255 0 0)
            buf[0 * 3 + 2, y] = 255;
            buf[0 * 3 + 1, y] = 0;
            buf[0 * 3 + 0, y] = 0;
            // Set the second pixel to 128 (gray)
            buf[1 * 3 + 2, y] = 0;
            buf[1 * 3 + 1, y] = 255;
            buf[1 * 3 + 0, y] = 0;
            // Set the third pixel to 255 (white)
            buf[2 * 3 + 2, y] = 0;
            buf[2 * 3 + 1, y] = 0;
            buf[2 * 3 + 0, y] = 255;

            TIS.Imaging.FrameExtensions.SaveAsBitmap(frame, "RGB24.bmp");
//>>
        }

        private void cmdRGB32_Click(object sender, EventArgs e)
        {
            TIS.Imaging.IFrameQueueBuffer frame = GrabImage(TIS.Imaging.MediaSubtypes.RGB32);
            if( frame == null ) return;

            BufferAccessHelper buf = new BufferAccessHelper( frame );

            //<<rgb32print
            // RGB32is bottom-up, the first line has index lines-1
            int y = frame.FrameType.Height - 1;

            txtOutput.Text = "Image buffer pixel format is RGB32\r\n";
            txtOutput.Text += "Pixel 1: ";
            txtOutput.Text += "R=" + buf[0 * 4 + 2, y] + ", ";
            txtOutput.Text += "G=" + buf[0 * 4 + 1, y] + ", ";
            txtOutput.Text += "B=" + buf[0 * 4 + 0, y] + "\r\n";
            txtOutput.Text += "Pixel 2: ";
            txtOutput.Text += "R=" + buf[1 * 4 + 2, y] + ", ";
            txtOutput.Text += "G=" + buf[1 * 4 + 1, y] + ", ";
            txtOutput.Text += "B=" + buf[1 * 4 + 0, y];
//>>

//<<rgb32edit
            // Set the first pixel to red (255 0 0)
            buf[0 * 4 + 2, y] = 255;
            buf[0 * 4 + 1, y] = 0;
            buf[0 * 4 + 0, y] = 0;
            // Set the second pixel to 128 (gray)
            buf[1 * 4 + 2, y] = 0;
            buf[1 * 4 + 1, y] = 255;
            buf[1 * 4 + 0, y] = 0;
            // Set the third pixel to 255 (white)
            buf[2 * 4 + 2, y] = 0;
            buf[2 * 4 + 1, y] = 0;
            buf[2 * 4 + 0, y] = 255;

            TIS.Imaging.FrameExtensions.SaveAsBitmap(frame, "RGB32.bmp");
            //>>
        }

        //<<y16read
        private unsafe UInt16 ReadY16(TIS.Imaging.IFrameQueueBuffer buf, int row, int col)
		{
			// Y16 is top-down, the first line has index 0
			int offset = row * buf.FrameType.BytesPerLine + col * 2;

			return (UInt16)System.Runtime.InteropServices.Marshal.ReadInt16(new IntPtr(buf.Ptr), offset);
		}
//>>

//<<y16write
		private unsafe void WriteY16(TIS.Imaging.IFrameQueueBuffer buf, int row, int col, UInt16 value)
		{
			int offset = row * buf.FrameType.BytesPerLine + col * 2;

			System.Runtime.InteropServices.Marshal.WriteInt16( new IntPtr( buf.Ptr ), offset, (short)value);
		}
//>>

		private void cmdY16_Click(object sender, EventArgs e)
		{
			TIS.Imaging.IFrameQueueBuffer buf = GrabImage(TIS.Imaging.MediaSubtypes.Y16);
			if (buf == null) return;

//<<y16print			
			UInt32 val0 = ReadY16(buf, 0, 0);
			UInt32 val1 = ReadY16(buf, 0, 1);

			txtOutput.Text = "Image buffer pixel format is Y16\r\n";
			txtOutput.Text += "Pixel 1: " + val0 + "\r\n";
			txtOutput.Text += "Pixel 2: " + val1;
//>>

//<<y16edit
			WriteY16(buf, 0, 0, 0x0000); // Black
			WriteY16(buf, 0, 1, 0x8000); // Gray
			WriteY16(buf, 0, 2, 0xFFFF); // White

            TIS.Imaging.FrameExtensions.SaveAsTiff( buf, "y16.tiff");
//>>
		}
    }
}