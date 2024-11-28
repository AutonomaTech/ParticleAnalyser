namespace Capturing_a_Video_File
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
			this.btnStartLive = new System.Windows.Forms.Button();
			this.btnStopLive = new System.Windows.Forms.Button();
			this.btnCaptureVideo = new System.Windows.Forms.Button();
			this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
			this.SuspendLayout();
			// 
			// btnStartLive
			// 
			this.btnStartLive.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
			this.btnStartLive.Location = new System.Drawing.Point(8, 200);
			this.btnStartLive.Name = "btnStartLive";
			this.btnStartLive.Size = new System.Drawing.Size(86, 23);
			this.btnStartLive.TabIndex = 1;
			this.btnStartLive.Text = "Start Live";
			this.btnStartLive.UseVisualStyleBackColor = true;
			this.btnStartLive.Click += new System.EventHandler(this.btnStartLive_Click);
			// 
			// btnStopLive
			// 
			this.btnStopLive.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
			this.btnStopLive.Location = new System.Drawing.Point(103, 200);
			this.btnStopLive.Name = "btnStopLive";
			this.btnStopLive.Size = new System.Drawing.Size(86, 23);
			this.btnStopLive.TabIndex = 2;
			this.btnStopLive.Text = "Stop Live";
			this.btnStopLive.UseVisualStyleBackColor = true;
			this.btnStopLive.Click += new System.EventHandler(this.btnStopLive_Click);
			// 
			// btnCaptureVideo
			// 
			this.btnCaptureVideo.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
			this.btnCaptureVideo.Location = new System.Drawing.Point(198, 200);
			this.btnCaptureVideo.Name = "btnCaptureVideo";
			this.btnCaptureVideo.Size = new System.Drawing.Size(86, 23);
			this.btnCaptureVideo.TabIndex = 3;
			this.btnCaptureVideo.Text = "Capture Video";
			this.btnCaptureVideo.UseVisualStyleBackColor = true;
			this.btnCaptureVideo.Click += new System.EventHandler(this.btnCaptureVideo_Click);
			// 
			// icImagingControl1
			// 
			this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
						| System.Windows.Forms.AnchorStyles.Left)
						| System.Windows.Forms.AnchorStyles.Right)));
			this.icImagingControl1.BackColor = System.Drawing.Color.White;
			this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
			this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
			this.icImagingControl1.Location = new System.Drawing.Point(10, 12);
			this.icImagingControl1.Name = "icImagingControl1";
			this.icImagingControl1.Size = new System.Drawing.Size(273, 177);
			this.icImagingControl1.TabIndex = 4;
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(296, 230);
			this.Controls.Add(this.icImagingControl1);
			this.Controls.Add(this.btnCaptureVideo);
			this.Controls.Add(this.btnStopLive);
			this.Controls.Add(this.btnStartLive);
			this.Name = "Form1";
			this.Text = "Capturing a Video File";
			this.Load += new System.EventHandler(this.Form1_Load);
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
			this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.Button btnStartLive;
        private System.Windows.Forms.Button btnStopLive;
        private System.Windows.Forms.Button btnCaptureVideo;
		private TIS.Imaging.ICImagingControl icImagingControl1;
    }
}

