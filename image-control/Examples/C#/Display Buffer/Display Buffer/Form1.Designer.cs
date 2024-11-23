namespace Display_Buffer
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
			this.cmdStart = new System.Windows.Forms.Button();
			this.cmdStop = new System.Windows.Forms.Button();
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
			this.icImagingControl1.Location = new System.Drawing.Point(12, 43);
			this.icImagingControl1.Name = "icImagingControl1";
			this.icImagingControl1.Size = new System.Drawing.Size(553, 382);
			this.icImagingControl1.TabIndex = 0;
			// 
			// cmdStart
			// 
			this.cmdStart.Location = new System.Drawing.Point(12, 12);
			this.cmdStart.Name = "cmdStart";
			this.cmdStart.Size = new System.Drawing.Size(75, 23);
			this.cmdStart.TabIndex = 1;
			this.cmdStart.Text = "Start";
			this.cmdStart.UseVisualStyleBackColor = true;
			this.cmdStart.Click += new System.EventHandler(this.cmdStart_Click);
			// 
			// cmdStop
			// 
			this.cmdStop.Location = new System.Drawing.Point(93, 12);
			this.cmdStop.Name = "cmdStop";
			this.cmdStop.Size = new System.Drawing.Size(75, 23);
			this.cmdStop.TabIndex = 2;
			this.cmdStop.Text = "Stop";
			this.cmdStop.UseVisualStyleBackColor = true;
			this.cmdStop.Click += new System.EventHandler(this.cmdStop_Click);
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(576, 437);
			this.Controls.Add(this.cmdStop);
			this.Controls.Add(this.cmdStart);
			this.Controls.Add(this.icImagingControl1);
			this.MinimumSize = new System.Drawing.Size(584, 471);
			this.Name = "Form1";
			this.Text = "Display Buffer";
			this.Load += new System.EventHandler(this.Form1_Load);
			this.SizeChanged += new System.EventHandler(this.Form1_SizeChanged);
			((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
			this.ResumeLayout(false);

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Button cmdStart;
        private System.Windows.Forms.Button cmdStop;
    }
}

