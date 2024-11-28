namespace Creating_an_Overlay
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
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
			this.components = new System.ComponentModel.Container();
			this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
			this.cmdDevice = new System.Windows.Forms.Button();
			this.cmdSettings = new System.Windows.Forms.Button();
			this.cmdStartStop = new System.Windows.Forms.Button();
			this.chkPPDevice = new System.Windows.Forms.CheckBox();
			this.chkPPSink = new System.Windows.Forms.CheckBox();
			this.chkPPDisplay = new System.Windows.Forms.CheckBox();
			this.btnColor = new System.Windows.Forms.RadioButton();
			this.btnGrayscale = new System.Windows.Forms.RadioButton();
			this.btnBestFit = new System.Windows.Forms.RadioButton();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
			this.SuspendLayout();
			// 
			// icImagingControl1
			// 
			this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
						| System.Windows.Forms.AnchorStyles.Left)
						| System.Windows.Forms.AnchorStyles.Right)));
			this.icImagingControl1.BackColor = System.Drawing.Color.White;
			this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
			this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
			this.icImagingControl1.Location = new System.Drawing.Point(8, 86);
			this.icImagingControl1.Name = "icImagingControl1";
			this.icImagingControl1.Size = new System.Drawing.Size(640, 480);
			this.icImagingControl1.TabIndex = 0;
			this.icImagingControl1.LivePrepared += new System.EventHandler(this.icImagingControl1_LivePrepared);
			this.icImagingControl1.OverlayUpdate += new System.EventHandler<TIS.Imaging.ICImagingControl.OverlayUpdateEventArgs>(this.icImagingControl1_OverlayUpdate);
			// 
			// cmdDevice
			// 
			this.cmdDevice.Location = new System.Drawing.Point(12, 12);
			this.cmdDevice.Name = "cmdDevice";
			this.cmdDevice.Size = new System.Drawing.Size(75, 23);
			this.cmdDevice.TabIndex = 1;
			this.cmdDevice.Text = "Device";
			this.cmdDevice.UseVisualStyleBackColor = true;
			this.cmdDevice.Click += new System.EventHandler(this.cmdDevice_Click);
			// 
			// cmdSettings
			// 
			this.cmdSettings.Location = new System.Drawing.Point(93, 12);
			this.cmdSettings.Name = "cmdSettings";
			this.cmdSettings.Size = new System.Drawing.Size(75, 23);
			this.cmdSettings.TabIndex = 2;
			this.cmdSettings.Text = "Settings";
			this.cmdSettings.UseVisualStyleBackColor = true;
			this.cmdSettings.Click += new System.EventHandler(this.cmdSettings_Click);
			// 
			// cmdStartStop
			// 
			this.cmdStartStop.Location = new System.Drawing.Point(174, 12);
			this.cmdStartStop.Name = "cmdStartStop";
			this.cmdStartStop.Size = new System.Drawing.Size(75, 23);
			this.cmdStartStop.TabIndex = 3;
			this.cmdStartStop.Text = "Start";
			this.cmdStartStop.UseVisualStyleBackColor = true;
			this.cmdStartStop.Click += new System.EventHandler(this.cmdStartStop_Click);
			// 
			// chkPPDevice
			// 
			this.chkPPDevice.AutoSize = true;
			this.chkPPDevice.Location = new System.Drawing.Point(367, 16);
			this.chkPPDevice.Name = "chkPPDevice";
			this.chkPPDevice.Size = new System.Drawing.Size(60, 17);
			this.chkPPDevice.TabIndex = 4;
			this.chkPPDevice.Text = "Device";
			this.chkPPDevice.UseVisualStyleBackColor = true;
			this.chkPPDevice.CheckedChanged += new System.EventHandler(this.chkPPDevice_CheckedChanged);
			// 
			// chkPPSink
			// 
			this.chkPPSink.AutoSize = true;
			this.chkPPSink.Location = new System.Drawing.Point(367, 39);
			this.chkPPSink.Name = "chkPPSink";
			this.chkPPSink.Size = new System.Drawing.Size(47, 17);
			this.chkPPSink.TabIndex = 5;
			this.chkPPSink.Text = "Sink";
			this.chkPPSink.UseVisualStyleBackColor = true;
			this.chkPPSink.CheckedChanged += new System.EventHandler(this.chkPPSink_CheckedChanged);
			// 
			// chkPPDisplay
			// 
			this.chkPPDisplay.AutoSize = true;
			this.chkPPDisplay.Location = new System.Drawing.Point(367, 62);
			this.chkPPDisplay.Name = "chkPPDisplay";
			this.chkPPDisplay.Size = new System.Drawing.Size(60, 17);
			this.chkPPDisplay.TabIndex = 6;
			this.chkPPDisplay.Text = "Display";
			this.chkPPDisplay.UseVisualStyleBackColor = true;
			this.chkPPDisplay.CheckedChanged += new System.EventHandler(this.chkPPDisplay_CheckedChanged);
			// 
			// btnColor
			// 
			this.btnColor.AutoSize = true;
			this.btnColor.Location = new System.Drawing.Point(559, 16);
			this.btnColor.Name = "btnColor";
			this.btnColor.Size = new System.Drawing.Size(49, 17);
			this.btnColor.TabIndex = 7;
			this.btnColor.TabStop = true;
			this.btnColor.Text = "Color";
			this.btnColor.UseVisualStyleBackColor = true;
			this.btnColor.CheckedChanged += new System.EventHandler(this.btnColor_CheckedChanged);
			// 
			// btnGrayscale
			// 
			this.btnGrayscale.AutoSize = true;
			this.btnGrayscale.Location = new System.Drawing.Point(559, 38);
			this.btnGrayscale.Name = "btnGrayscale";
			this.btnGrayscale.Size = new System.Drawing.Size(72, 17);
			this.btnGrayscale.TabIndex = 8;
			this.btnGrayscale.TabStop = true;
			this.btnGrayscale.Text = "Grayscale";
			this.btnGrayscale.UseVisualStyleBackColor = true;
			this.btnGrayscale.CheckedChanged += new System.EventHandler(this.btnGrayscale_CheckedChanged);
			// 
			// btnBestFit
			// 
			this.btnBestFit.AutoSize = true;
			this.btnBestFit.Location = new System.Drawing.Point(559, 62);
			this.btnBestFit.Name = "btnBestFit";
			this.btnBestFit.Size = new System.Drawing.Size(60, 17);
			this.btnBestFit.TabIndex = 9;
			this.btnBestFit.TabStop = true;
			this.btnBestFit.Text = "Best Fit";
			this.btnBestFit.UseVisualStyleBackColor = true;
			this.btnBestFit.CheckedChanged += new System.EventHandler(this.btnBestFit_CheckedChanged);
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(283, 18);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(72, 13);
			this.label1.TabIndex = 10;
			this.label1.Text = "Path Position:";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(483, 18);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(64, 13);
			this.label2.TabIndex = 11;
			this.label2.Text = "Color Mode:";
			// 
			// timer1
			// 
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(655, 572);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.btnBestFit);
			this.Controls.Add(this.btnGrayscale);
			this.Controls.Add(this.btnColor);
			this.Controls.Add(this.chkPPDisplay);
			this.Controls.Add(this.chkPPSink);
			this.Controls.Add(this.chkPPDevice);
			this.Controls.Add(this.cmdStartStop);
			this.Controls.Add(this.cmdSettings);
			this.Controls.Add(this.cmdDevice);
			this.Controls.Add(this.icImagingControl1);
			this.MinimumSize = new System.Drawing.Size(663, 606);
			this.Name = "Form1";
			this.Text = "Creating an Overlay";
			this.Load += new System.EventHandler(this.Form1_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button cmdDevice;
        private System.Windows.Forms.Button cmdSettings;
        private System.Windows.Forms.Button cmdStartStop;
        private System.Windows.Forms.CheckBox chkPPDevice;
        private System.Windows.Forms.CheckBox chkPPSink;
        private System.Windows.Forms.CheckBox chkPPDisplay;
        private System.Windows.Forms.RadioButton btnColor;
        private System.Windows.Forms.RadioButton btnGrayscale;
        private System.Windows.Forms.RadioButton btnBestFit;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Timer timer1;
    }
}

