namespace MakingDeviceSettings
{
	partial class Form1
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose( bool disposing )
		{
			if( disposing && ( components != null ) )
			{
				components.Dispose();
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
			this.cmdStartLive = new System.Windows.Forms.Button();
			this.cmdStopLive = new System.Windows.Forms.Button();
			this.cmdDevice = new System.Windows.Forms.Button();
			( (System.ComponentModel.ISupportInitialize)( this.icImagingControl1 ) ).BeginInit();
			this.SuspendLayout();
			// 
			// icImagingControl1
			// 
			this.icImagingControl1.Anchor = ( (System.Windows.Forms.AnchorStyles)( ( ( ( System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom )
						| System.Windows.Forms.AnchorStyles.Left )
						| System.Windows.Forms.AnchorStyles.Right ) ) );
			this.icImagingControl1.BackColor = System.Drawing.Color.White;
			this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
			this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point( 0, 0 );
			this.icImagingControl1.Location = new System.Drawing.Point( 12, 12 );
			this.icImagingControl1.Name = "icImagingControl1";
			this.icImagingControl1.Size = new System.Drawing.Size( 368, 312 );
			this.icImagingControl1.TabIndex = 0;
			// 
			// cmdStartLive
			// 
			this.cmdStartLive.Anchor = ( (System.Windows.Forms.AnchorStyles)( ( System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left ) ) );
			this.cmdStartLive.Location = new System.Drawing.Point( 12, 330 );
			this.cmdStartLive.Name = "cmdStartLive";
			this.cmdStartLive.Size = new System.Drawing.Size( 75, 23 );
			this.cmdStartLive.TabIndex = 1;
			this.cmdStartLive.Text = "Start Live";
			this.cmdStartLive.UseVisualStyleBackColor = true;
			this.cmdStartLive.Click += new System.EventHandler( this.cmdStartLive_Click );
			// 
			// cmdStopLive
			// 
			this.cmdStopLive.Anchor = ( (System.Windows.Forms.AnchorStyles)( ( System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left ) ) );
			this.cmdStopLive.Location = new System.Drawing.Point( 93, 330 );
			this.cmdStopLive.Name = "cmdStopLive";
			this.cmdStopLive.Size = new System.Drawing.Size( 75, 23 );
			this.cmdStopLive.TabIndex = 2;
			this.cmdStopLive.Text = "Stop Live";
			this.cmdStopLive.UseVisualStyleBackColor = true;
			this.cmdStopLive.Click += new System.EventHandler( this.cmdStopLive_Click );
			// 
			// cmdDevice
			// 
			this.cmdDevice.Anchor = ( (System.Windows.Forms.AnchorStyles)( ( System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right ) ) );
			this.cmdDevice.Location = new System.Drawing.Point( 305, 330 );
			this.cmdDevice.Name = "cmdDevice";
			this.cmdDevice.Size = new System.Drawing.Size( 75, 23 );
			this.cmdDevice.TabIndex = 3;
			this.cmdDevice.Text = "Device";
			this.cmdDevice.UseVisualStyleBackColor = true;
			this.cmdDevice.Click += new System.EventHandler( this.cmdDevice_Click );
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF( 6F, 13F );
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size( 392, 365 );
			this.Controls.Add( this.cmdDevice );
			this.Controls.Add( this.cmdStopLive );
			this.Controls.Add( this.cmdStartLive );
			this.Controls.Add( this.icImagingControl1 );
			this.Name = "Form1";
			this.Text = "Making Device Settings";
			this.Load += new System.EventHandler( this.Form1_Load );
			( (System.ComponentModel.ISupportInitialize)( this.icImagingControl1 ) ).EndInit();
			this.ResumeLayout( false );

		}

		#endregion

		private TIS.Imaging.ICImagingControl icImagingControl1;
		private System.Windows.Forms.Button cmdStartLive;
		private System.Windows.Forms.Button cmdStopLive;
		private System.Windows.Forms.Button cmdDevice;
	}
}

