namespace Filter_Inspector
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
			this.btnStartLive = new System.Windows.Forms.Button();
			this.btnProperties = new System.Windows.Forms.Button();
			this.btnStopLive = new System.Windows.Forms.Button();
			this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
			this.label1 = new System.Windows.Forms.Label();
			this.lstFrameFilterModules = new System.Windows.Forms.ListBox();
			this.lstFrameFilters = new System.Windows.Forms.ListBox();
			this.label2 = new System.Windows.Forms.Label();
			this.groupBox1 = new System.Windows.Forms.GroupBox();
			this.btnRemove = new System.Windows.Forms.Button();
			this.btnDialog = new System.Windows.Forms.Button();
			this.lblSelectedFilter = new System.Windows.Forms.Label();
			( (System.ComponentModel.ISupportInitialize)( this.icImagingControl1 ) ).BeginInit();
			this.groupBox1.SuspendLayout();
			this.SuspendLayout();
			// 
			// btnDevice
			// 
			this.btnDevice.Location = new System.Drawing.Point( 12, 12 );
			this.btnDevice.Name = "btnDevice";
			this.btnDevice.Size = new System.Drawing.Size( 75, 23 );
			this.btnDevice.TabIndex = 0;
			this.btnDevice.Text = "Device";
			this.btnDevice.UseVisualStyleBackColor = true;
			this.btnDevice.Click += new System.EventHandler( this.btnDevice_Click );
			// 
			// btnStartLive
			// 
			this.btnStartLive.Location = new System.Drawing.Point( 12, 41 );
			this.btnStartLive.Name = "btnStartLive";
			this.btnStartLive.Size = new System.Drawing.Size( 75, 23 );
			this.btnStartLive.TabIndex = 1;
			this.btnStartLive.Text = "Start Live";
			this.btnStartLive.UseVisualStyleBackColor = true;
			this.btnStartLive.Click += new System.EventHandler( this.btnStartLive_Click );
			// 
			// btnProperties
			// 
			this.btnProperties.Location = new System.Drawing.Point( 93, 12 );
			this.btnProperties.Name = "btnProperties";
			this.btnProperties.Size = new System.Drawing.Size( 75, 23 );
			this.btnProperties.TabIndex = 2;
			this.btnProperties.Text = "Properies";
			this.btnProperties.UseVisualStyleBackColor = true;
			this.btnProperties.Click += new System.EventHandler( this.btnProperties_Click );
			// 
			// btnStopLive
			// 
			this.btnStopLive.Location = new System.Drawing.Point( 93, 41 );
			this.btnStopLive.Name = "btnStopLive";
			this.btnStopLive.Size = new System.Drawing.Size( 75, 23 );
			this.btnStopLive.TabIndex = 3;
			this.btnStopLive.Text = "Stop Live";
			this.btnStopLive.UseVisualStyleBackColor = true;
			this.btnStopLive.Click += new System.EventHandler( this.btnStopLive_Click );
			// 
			// icImagingControl1
			// 
			this.icImagingControl1.Anchor = ( (System.Windows.Forms.AnchorStyles)( ( ( ( System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom )
						| System.Windows.Forms.AnchorStyles.Left )
						| System.Windows.Forms.AnchorStyles.Right ) ) );
			this.icImagingControl1.BackColor = System.Drawing.Color.White;
			this.icImagingControl1.ImageAvailableExecutionMode = TIS.Imaging.EventExecutionMode.MultiThreaded;
			this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point( 0, 0 );
			this.icImagingControl1.Location = new System.Drawing.Point( 176, 14 );
			this.icImagingControl1.Name = "icImagingControl1";
			this.icImagingControl1.Size = new System.Drawing.Size( 509, 442 );
			this.icImagingControl1.TabIndex = 4;
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point( 16, 78 );
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size( 104, 13 );
			this.label1.TabIndex = 5;
			this.label1.Text = "Frame Filter Modules";
			// 
			// lstFrameFilterModules
			// 
			this.lstFrameFilterModules.FormattingEnabled = true;
			this.lstFrameFilterModules.Location = new System.Drawing.Point( 12, 94 );
			this.lstFrameFilterModules.Name = "lstFrameFilterModules";
			this.lstFrameFilterModules.Size = new System.Drawing.Size( 156, 95 );
			this.lstFrameFilterModules.TabIndex = 6;
			this.lstFrameFilterModules.SelectedIndexChanged += new System.EventHandler( this.lstFrameFilterModules_SelectedIndexChanged );
			// 
			// lstFrameFilters
			// 
			this.lstFrameFilters.FormattingEnabled = true;
			this.lstFrameFilters.Location = new System.Drawing.Point( 12, 219 );
			this.lstFrameFilters.Name = "lstFrameFilters";
			this.lstFrameFilters.Size = new System.Drawing.Size( 156, 95 );
			this.lstFrameFilters.TabIndex = 8;
			this.lstFrameFilters.SelectedIndexChanged += new System.EventHandler( this.lstFrameFilters_SelectedIndexChanged );
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point( 16, 203 );
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size( 66, 13 );
			this.label2.TabIndex = 7;
			this.label2.Text = "Frame Filters";
			// 
			// groupBox1
			// 
			this.groupBox1.Controls.Add( this.btnRemove );
			this.groupBox1.Controls.Add( this.btnDialog );
			this.groupBox1.Controls.Add( this.lblSelectedFilter );
			this.groupBox1.Location = new System.Drawing.Point( 16, 324 );
			this.groupBox1.Name = "groupBox1";
			this.groupBox1.Size = new System.Drawing.Size( 151, 131 );
			this.groupBox1.TabIndex = 9;
			this.groupBox1.TabStop = false;
			this.groupBox1.Text = "groupBox1";
			// 
			// btnRemove
			// 
			this.btnRemove.Location = new System.Drawing.Point( 6, 88 );
			this.btnRemove.Name = "btnRemove";
			this.btnRemove.Size = new System.Drawing.Size( 75, 23 );
			this.btnRemove.TabIndex = 3;
			this.btnRemove.Text = "Remove";
			this.btnRemove.UseVisualStyleBackColor = true;
			this.btnRemove.Click += new System.EventHandler( this.btnRemove_Click );
			// 
			// btnDialog
			// 
			this.btnDialog.Location = new System.Drawing.Point( 6, 59 );
			this.btnDialog.Name = "btnDialog";
			this.btnDialog.Size = new System.Drawing.Size( 75, 23 );
			this.btnDialog.TabIndex = 2;
			this.btnDialog.Text = "Dialog";
			this.btnDialog.UseVisualStyleBackColor = true;
			this.btnDialog.Click += new System.EventHandler( this.btnDialog_Click );
			// 
			// lblSelectedFilter
			// 
			this.lblSelectedFilter.AutoSize = true;
			this.lblSelectedFilter.Location = new System.Drawing.Point( 6, 26 );
			this.lblSelectedFilter.Name = "lblSelectedFilter";
			this.lblSelectedFilter.Size = new System.Drawing.Size( 35, 13 );
			this.lblSelectedFilter.TabIndex = 0;
			this.lblSelectedFilter.Text = "label3";
			// 
			// Form1
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF( 6F, 13F );
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size( 696, 469 );
			this.Controls.Add( this.groupBox1 );
			this.Controls.Add( this.lstFrameFilters );
			this.Controls.Add( this.label2 );
			this.Controls.Add( this.lstFrameFilterModules );
			this.Controls.Add( this.label1 );
			this.Controls.Add( this.icImagingControl1 );
			this.Controls.Add( this.btnStopLive );
			this.Controls.Add( this.btnProperties );
			this.Controls.Add( this.btnStartLive );
			this.Controls.Add( this.btnDevice );
			this.Name = "Form1";
			this.Text = "Filter Inspector";
			this.Load += new System.EventHandler( this.Form1_Load );
			( (System.ComponentModel.ISupportInitialize)( this.icImagingControl1 ) ).EndInit();
			this.groupBox1.ResumeLayout( false );
			this.groupBox1.PerformLayout();
			this.ResumeLayout( false );
			this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnDevice;
        private System.Windows.Forms.Button btnStartLive;
        private System.Windows.Forms.Button btnProperties;
        private System.Windows.Forms.Button btnStopLive;
        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ListBox lstFrameFilterModules;
        private System.Windows.Forms.ListBox lstFrameFilters;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnRemove;
        private System.Windows.Forms.Button btnDialog;
        private System.Windows.Forms.Label lblSelectedFilter;
    }
}

