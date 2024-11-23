namespace Image_Processing
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
            this.cmdStartLive = new System.Windows.Forms.Button();
            this.cmdStopLive = new System.Windows.Forms.Button();
            this.cmdProcess = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(9, 9);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(640, 480);
            this.icImagingControl1.TabIndex = 0;
            // 
            // cmdStartLive
            // 
            this.cmdStartLive.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cmdStartLive.Location = new System.Drawing.Point(12, 495);
            this.cmdStartLive.Name = "cmdStartLive";
            this.cmdStartLive.Size = new System.Drawing.Size(75, 23);
            this.cmdStartLive.TabIndex = 1;
            this.cmdStartLive.Text = "Start Live";
            this.cmdStartLive.UseVisualStyleBackColor = true;
            this.cmdStartLive.Click += new System.EventHandler(this.cmdStartLive_Click_1);
            // 
            // cmdStopLive
            // 
            this.cmdStopLive.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cmdStopLive.Location = new System.Drawing.Point(93, 495);
            this.cmdStopLive.Name = "cmdStopLive";
            this.cmdStopLive.Size = new System.Drawing.Size(75, 23);
            this.cmdStopLive.TabIndex = 2;
            this.cmdStopLive.Text = "Stop Live";
            this.cmdStopLive.UseVisualStyleBackColor = true;
            this.cmdStopLive.Click += new System.EventHandler(this.cmdStopLive_Click);
            // 
            // cmdProcess
            // 
            this.cmdProcess.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.cmdProcess.Location = new System.Drawing.Point(174, 495);
            this.cmdProcess.Name = "cmdProcess";
            this.cmdProcess.Size = new System.Drawing.Size(75, 23);
            this.cmdProcess.TabIndex = 3;
            this.cmdProcess.Text = "Process";
            this.cmdProcess.UseVisualStyleBackColor = true;
            this.cmdProcess.Click += new System.EventHandler(this.cmdProcess_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(659, 528);
            this.Controls.Add(this.cmdProcess);
            this.Controls.Add(this.cmdStopLive);
            this.Controls.Add(this.cmdStartLive);
            this.Controls.Add(this.icImagingControl1);
            this.MinimumSize = new System.Drawing.Size(667, 562);
            this.Name = "Form1";
            this.Text = "Performing Image Processing";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button cmdStartLive;
        private System.Windows.Forms.Button cmdStopLive;
        private System.Windows.Forms.Button cmdProcess;
    }
}

