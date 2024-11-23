namespace Capturing_an_AVI_File
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
            this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
            this.cmdCaptureAVI = new System.Windows.Forms.Button();
            this.cmdStartLive = new System.Windows.Forms.Button();
            this.cmdStopLive = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
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
            this.icImagingControl1.Location = new System.Drawing.Point(22, 19);
            this.icImagingControl1.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(648, 323);
            this.icImagingControl1.TabIndex = 0;
            // 
            // cmdCaptureAVI
            // 
            this.cmdCaptureAVI.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cmdCaptureAVI.Location = new System.Drawing.Point(489, 354);
            this.cmdCaptureAVI.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdCaptureAVI.Name = "cmdCaptureAVI";
            this.cmdCaptureAVI.Size = new System.Drawing.Size(184, 44);
            this.cmdCaptureAVI.TabIndex = 2;
            this.cmdCaptureAVI.Text = "Capture AVI";
            this.cmdCaptureAVI.UseVisualStyleBackColor = true;
            this.cmdCaptureAVI.Click += new System.EventHandler(this.cmdCaptureAVI_Click);
            // 
            // cmdStartLive
            // 
            this.cmdStartLive.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cmdStartLive.Location = new System.Drawing.Point(0, 354);
            this.cmdStartLive.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdStartLive.Name = "cmdStartLive";
            this.cmdStartLive.Size = new System.Drawing.Size(184, 44);
            this.cmdStartLive.TabIndex = 3;
            this.cmdStartLive.Text = "Start Live";
            this.cmdStartLive.UseVisualStyleBackColor = true;
            this.cmdStartLive.Click += new System.EventHandler(this.cmdStartLive_Click);
            // 
            // cmdStopLive
            // 
            this.cmdStopLive.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cmdStopLive.Location = new System.Drawing.Point(246, 354);
            this.cmdStopLive.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.cmdStopLive.Name = "cmdStopLive";
            this.cmdStopLive.Size = new System.Drawing.Size(184, 44);
            this.cmdStopLive.TabIndex = 4;
            this.cmdStopLive.Text = "Stop Live";
            this.cmdStopLive.UseVisualStyleBackColor = true;
            this.cmdStopLive.Click += new System.EventHandler(this.cmdStopLive_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(12F, 25F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(688, 410);
            this.Controls.Add(this.cmdStopLive);
            this.Controls.Add(this.cmdStartLive);
            this.Controls.Add(this.cmdCaptureAVI);
            this.Controls.Add(this.icImagingControl1);
            this.Margin = new System.Windows.Forms.Padding(6, 6, 6, 6);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button cmdCaptureAVI;
        private System.Windows.Forms.Button cmdStartLive;
        private System.Windows.Forms.Button cmdStopLive;
    }
}

