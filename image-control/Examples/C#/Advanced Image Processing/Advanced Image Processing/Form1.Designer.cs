namespace Advanced_Image_Processing
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
            this.cmdStart = new System.Windows.Forms.Button();
            this.cmdDevice = new System.Windows.Forms.Button();
            this.cmdStop = new System.Windows.Forms.Button();
            this.cmdSettings = new System.Windows.Forms.Button();
            this.cmdROICommit = new System.Windows.Forms.Button();
            this.sldThresholdSlider = new System.Windows.Forms.TrackBar();
            this.label1 = new System.Windows.Forms.Label();
            this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
            ((System.ComponentModel.ISupportInitialize)(this.sldThresholdSlider)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // cmdStart
            // 
            this.cmdStart.Location = new System.Drawing.Point(14, 23);
            this.cmdStart.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdStart.Name = "cmdStart";
            this.cmdStart.Size = new System.Drawing.Size(150, 44);
            this.cmdStart.TabIndex = 1;
            this.cmdStart.Text = "Start";
            this.cmdStart.UseVisualStyleBackColor = true;
            this.cmdStart.Click += new System.EventHandler(this.cmdStart_Click);
            // 
            // cmdDevice
            // 
            this.cmdDevice.Location = new System.Drawing.Point(14, 79);
            this.cmdDevice.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdDevice.Name = "cmdDevice";
            this.cmdDevice.Size = new System.Drawing.Size(150, 44);
            this.cmdDevice.TabIndex = 2;
            this.cmdDevice.Text = "Device";
            this.cmdDevice.UseVisualStyleBackColor = true;
            this.cmdDevice.Click += new System.EventHandler(this.cmdDevice_Click);
            // 
            // cmdStop
            // 
            this.cmdStop.Location = new System.Drawing.Point(176, 23);
            this.cmdStop.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdStop.Name = "cmdStop";
            this.cmdStop.Size = new System.Drawing.Size(150, 44);
            this.cmdStop.TabIndex = 3;
            this.cmdStop.Text = "Stop";
            this.cmdStop.UseVisualStyleBackColor = true;
            this.cmdStop.Click += new System.EventHandler(this.cmdStop_Click);
            // 
            // cmdSettings
            // 
            this.cmdSettings.Location = new System.Drawing.Point(176, 79);
            this.cmdSettings.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdSettings.Name = "cmdSettings";
            this.cmdSettings.Size = new System.Drawing.Size(150, 44);
            this.cmdSettings.TabIndex = 4;
            this.cmdSettings.Text = "Settings";
            this.cmdSettings.UseVisualStyleBackColor = true;
            this.cmdSettings.Click += new System.EventHandler(this.cmdSettings_Click);
            // 
            // cmdROICommit
            // 
            this.cmdROICommit.Location = new System.Drawing.Point(338, 23);
            this.cmdROICommit.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdROICommit.Name = "cmdROICommit";
            this.cmdROICommit.Size = new System.Drawing.Size(212, 44);
            this.cmdROICommit.TabIndex = 5;
            this.cmdROICommit.Text = "Set Current ROI";
            this.cmdROICommit.UseVisualStyleBackColor = true;
            this.cmdROICommit.Click += new System.EventHandler(this.cmdROICommit_Click);
            // 
            // sldThresholdSlider
            // 
            this.sldThresholdSlider.Location = new System.Drawing.Point(562, 23);
            this.sldThresholdSlider.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.sldThresholdSlider.Maximum = 100;
            this.sldThresholdSlider.Name = "sldThresholdSlider";
            this.sldThresholdSlider.Size = new System.Drawing.Size(564, 90);
            this.sldThresholdSlider.TabIndex = 6;
            this.sldThresholdSlider.Scroll += new System.EventHandler(this.sldThresholdSlider_Scroll);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(790, 98);
            this.label1.Margin = new System.Windows.Forms.Padding(6, 0, 6, 0);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(108, 25);
            this.label1.TabIndex = 7;
            this.label1.Text = "Threshold";
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.DeviceListChangedExecutionMode = TIS.Imaging.EventExecutionMode.Invoke;
            this.icImagingControl1.DeviceLostExecutionMode = TIS.Imaging.EventExecutionMode.AsyncInvoke;
            this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(26, 142);
            this.icImagingControl1.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(1098, 608);
            this.icImagingControl1.TabIndex = 8;
            this.icImagingControl1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.icImagingControl1_MouseDown);
            this.icImagingControl1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.icImagingControl1_MouseMove);
            this.icImagingControl1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.icImagingControl1_MouseUp);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(1150, 777);
            this.Controls.Add(this.icImagingControl1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.sldThresholdSlider);
            this.Controls.Add(this.cmdROICommit);
            this.Controls.Add(this.cmdSettings);
            this.Controls.Add(this.cmdStop);
            this.Controls.Add(this.cmdDevice);
            this.Controls.Add(this.cmdStart);
            this.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.Name = "Form1";
            this.Text = "Advanced Image Processing";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.sldThresholdSlider)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion


        private System.Windows.Forms.Button cmdStart;
        private System.Windows.Forms.Button cmdDevice;
        private System.Windows.Forms.Button cmdStop;
        private System.Windows.Forms.Button cmdSettings;
        private System.Windows.Forms.Button cmdROICommit;
        private System.Windows.Forms.TrackBar sldThresholdSlider;
        private System.Windows.Forms.Label label1;
		private TIS.Imaging.ICImagingControl icImagingControl1;
    }
}

