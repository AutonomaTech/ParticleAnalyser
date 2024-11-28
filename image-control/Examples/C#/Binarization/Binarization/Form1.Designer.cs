namespace Binarization
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
			this.btnDevice = new System.Windows.Forms.Button();
			this.btnProperties = new System.Windows.Forms.Button();
			this.chkEnable = new System.Windows.Forms.CheckBox();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.lblThreshold = new System.Windows.Forms.Label();
			this.sldThreshold = new System.Windows.Forms.TrackBar();
			this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
			((System.ComponentModel.ISupportInitialize)(this.sldThreshold)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
			this.SuspendLayout();
			// 
			// btnDevice
			// 
			this.btnDevice.Location = new System.Drawing.Point(480, 13);
			this.btnDevice.Name = "btnDevice";
			this.btnDevice.Size = new System.Drawing.Size(75, 23);
			this.btnDevice.TabIndex = 1;
			this.btnDevice.Text = "Device";
			this.btnDevice.UseVisualStyleBackColor = true;
			this.btnDevice.Click += new System.EventHandler(this.btnDevice_Click);
			// 
			// btnProperties
			// 
			this.btnProperties.Location = new System.Drawing.Point(480, 42);
			this.btnProperties.Name = "btnProperties";
			this.btnProperties.Size = new System.Drawing.Size(75, 23);
			this.btnProperties.TabIndex = 3;
			this.btnProperties.Text = "Properties";
			this.btnProperties.UseVisualStyleBackColor = true;
			this.btnProperties.Click += new System.EventHandler(this.btnProperties_Click);
			// 
			// chkEnable
			// 
			this.chkEnable.AutoSize = true;
			this.chkEnable.Location = new System.Drawing.Point(80, 381);
			this.chkEnable.Name = "chkEnable";
			this.chkEnable.Size = new System.Drawing.Size(59, 17);
			this.chkEnable.TabIndex = 4;
			this.chkEnable.Text = "Enable";
			this.chkEnable.UseVisualStyleBackColor = true;
			this.chkEnable.CheckedChanged += new System.EventHandler(this.chkEnable_CheckedChanged);
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point(13, 382);
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size(61, 13);
			this.label1.TabIndex = 5;
			this.label1.Text = "Binarization";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point(159, 382);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(54, 13);
			this.label2.TabIndex = 6;
			this.label2.Text = "Threshold";
			// 
			// lblThreshold
			// 
			this.lblThreshold.AutoSize = true;
			this.lblThreshold.Location = new System.Drawing.Point(461, 382);
			this.lblThreshold.Name = "lblThreshold";
			this.lblThreshold.Size = new System.Drawing.Size(13, 13);
			this.lblThreshold.TabIndex = 7;
			this.lblThreshold.Text = "0";
			// 
			// sldThreshold
			// 
			this.sldThreshold.Location = new System.Drawing.Point(219, 374);
			this.sldThreshold.Name = "sldThreshold";
			this.sldThreshold.Size = new System.Drawing.Size(236, 45);
			this.sldThreshold.TabIndex = 8;
			this.sldThreshold.TickFrequency = 3;
			this.sldThreshold.Scroll += new System.EventHandler(this.sldThreshold_Scroll);
			// 
			// icImagingControl1
			// 
			this.icImagingControl1.BackColor = System.Drawing.Color.White;
			this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
			this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
			this.icImagingControl1.Location = new System.Drawing.Point(12, 16);
			this.icImagingControl1.Name = "icImagingControl1";
			this.icImagingControl1.Size = new System.Drawing.Size(452, 346);
			this.icImagingControl1.TabIndex = 9;
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(564, 429);
			this.Controls.Add(this.icImagingControl1);
			this.Controls.Add(this.sldThreshold);
			this.Controls.Add(this.lblThreshold);
			this.Controls.Add(this.label2);
			this.Controls.Add(this.label1);
			this.Controls.Add(this.chkEnable);
			this.Controls.Add(this.btnProperties);
			this.Controls.Add(this.btnDevice);
			this.Name = "Form1";
			this.Text = "Frame Filter Demo - Binarization";
			this.Load += new System.EventHandler(this.Form1_Load);
			this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
			((System.ComponentModel.ISupportInitialize)(this.sldThreshold)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
			this.ResumeLayout(false);
			this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnDevice;
        private System.Windows.Forms.Button btnProperties;
        private System.Windows.Forms.CheckBox chkEnable;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label lblThreshold;
        private System.Windows.Forms.TrackBar sldThreshold;
		private TIS.Imaging.ICImagingControl icImagingControl1;
    }
}

