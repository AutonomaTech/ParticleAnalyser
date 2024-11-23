using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using TIS.Imaging;

namespace Display_Buffer
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Form_Load
        ///
        /// Initializes the buttons and sets the size of the control
        /// to the size of the currently selected video format.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
		//<<Form_Load_beg
        private void Form1_Load( object sender, EventArgs e )
        {
            if( !icImagingControl1.LoadShowSaveDeviceState("lastSelectedDeviceState.xml") )
            {
                MessageBox.Show("No device was selected.", this.Text, MessageBoxButtons.OK, MessageBoxIcon.Information);
                this.Close();
                return;
            }

            // Change display dimensions to stretch video to full control size
            icImagingControl1.LiveDisplayDefault = false;
            icImagingControl1.LiveDisplaySize = icImagingControl1.Size;

            cmdStop.Enabled = false;

            icImagingControl1.LiveDisplay = false;

            InitSink();
        }
        //>>

        //<<InitSink
        private void InitSink()
        {
            icImagingControl1.Sink = new FrameQueueSink(ShowBuffer, MediaSubtypes.RGB32, 5);
        }
        //>>InitSink

        /// <summary>
        /// cmdStart
        ///
        /// Starts the Display.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<cmdStart_beg
        private void cmdStart_Click( object sender, EventArgs e )
        {
            icImagingControl1.LiveStart();
            cmdStart.Enabled = false;
            cmdStop.Enabled = true;
        }
        //>>

        /// <summary>
        /// cmdStop
        ///
        /// Stops the Display.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<cmdStop_beg
        private void cmdStop_Click( object sender, EventArgs e )
        {
            cmdStart.Enabled = true;
            cmdStop.Enabled = false;
            icImagingControl1.LiveStop();
            icImagingControl1.DisplayImageBufferClear();
        }
        //>>

        /// <summary>
        /// ICImagingControl1_ImageAvailable
        ///
        /// Retrieves the buffer specified by BufferIndex
        /// from the collection and displays it in the control window.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        //<<ShowBuffer
        private FrameQueuedResult ShowBuffer( IFrameQueueBuffer buffer )
        {
            try
            {
                icImagingControl1.DisplayImageBuffer( buffer );
            }
            catch( Exception ex )
            {
                System.Diagnostics.Trace.WriteLine( ex.Message );
            }
            return FrameQueuedResult.ReQueue;
        }
        //>>

        private void Form1_SizeChanged( object sender, EventArgs e )
        {
            if( icImagingControl1.DeviceValid )
            {
                icImagingControl1.LiveDisplaySize = icImagingControl1.Size;
            }
        }


    }
}