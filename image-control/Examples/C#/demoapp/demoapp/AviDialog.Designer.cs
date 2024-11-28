namespace demoapp
{
    partial class AviDialog
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
            this.chkCapturePause = new System.Windows.Forms.CheckBox();
            this.cmdStopButton = new System.Windows.Forms.Button();
            this.cmdStartButton = new System.Windows.Forms.Button();
            this.cmdCloseButton = new System.Windows.Forms.Button();
            this.GroupBox1 = new System.Windows.Forms.GroupBox();
            this.Label1 = new System.Windows.Forms.Label();
            this.edtAVIFile = new System.Windows.Forms.TextBox();
            this.cmdAVIFileChooser = new System.Windows.Forms.Button();
            this.GroupBox2 = new System.Windows.Forms.GroupBox();
            this.Label3 = new System.Windows.Forms.Label();
            this.edtConfigFile = new System.Windows.Forms.TextBox();
            this.cmdCodecDataFileChooser = new System.Windows.Forms.Button();
            this.cmdSaveCodecData = new System.Windows.Forms.Button();
            this.cmdLoadCodecData = new System.Windows.Forms.Button();
            this.Label2 = new System.Windows.Forms.Label();
            this.cmdShowPropPage = new System.Windows.Forms.Button();
            this.CodecBox = new System.Windows.Forms.ComboBox();
            this.GroupBox1.SuspendLayout();
            this.GroupBox2.SuspendLayout();
            this.SuspendLayout();
            // 
            // chkCapturePause
            // 
            this.chkCapturePause.Location = new System.Drawing.Point(340, 180);
            this.chkCapturePause.Name = "chkCapturePause";
            this.chkCapturePause.Size = new System.Drawing.Size(64, 24);
            this.chkCapturePause.TabIndex = 17;
            this.chkCapturePause.Text = "Pause";
            this.chkCapturePause.Click += new System.EventHandler(this.chkCapturePause_CheckedChanged);
            // 
            // cmdStopButton
            // 
            this.cmdStopButton.Location = new System.Drawing.Point(196, 180);
            this.cmdStopButton.Name = "cmdStopButton";
            this.cmdStopButton.Size = new System.Drawing.Size(128, 24);
            this.cmdStopButton.TabIndex = 16;
            this.cmdStopButton.Text = "Stop";
            this.cmdStopButton.Click += new System.EventHandler(this.cmdStopButton_Click);
            // 
            // cmdStartButton
            // 
            this.cmdStartButton.ImageIndex = 0;
            this.cmdStartButton.Location = new System.Drawing.Point(60, 180);
            this.cmdStartButton.Name = "cmdStartButton";
            this.cmdStartButton.Size = new System.Drawing.Size(128, 24);
            this.cmdStartButton.TabIndex = 15;
            this.cmdStartButton.Text = "Start";
            this.cmdStartButton.Click += new System.EventHandler(this.cmdStartButton_Click);
            // 
            // cmdCloseButton
            // 
            this.cmdCloseButton.Location = new System.Drawing.Point(380, 220);
            this.cmdCloseButton.Name = "cmdCloseButton";
            this.cmdCloseButton.Size = new System.Drawing.Size(88, 24);
            this.cmdCloseButton.TabIndex = 14;
            this.cmdCloseButton.Text = "Close";
            this.cmdCloseButton.Click += new System.EventHandler(this.cmdCloseButton_Click);
            // 
            // GroupBox1
            // 
            this.GroupBox1.Controls.Add(this.Label1);
            this.GroupBox1.Controls.Add(this.edtAVIFile);
            this.GroupBox1.Controls.Add(this.cmdAVIFileChooser);
            this.GroupBox1.Location = new System.Drawing.Point(12, 132);
            this.GroupBox1.Name = "GroupBox1";
            this.GroupBox1.Size = new System.Drawing.Size(456, 80);
            this.GroupBox1.TabIndex = 18;
            this.GroupBox1.TabStop = false;
            this.GroupBox1.Text = "Avi Capture Control";
            // 
            // Label1
            // 
            this.Label1.Location = new System.Drawing.Point(8, 18);
            this.Label1.Name = "Label1";
            this.Label1.Size = new System.Drawing.Size(64, 16);
            this.Label1.TabIndex = 3;
            this.Label1.Text = "File name:";
            // 
            // edtAVIFile
            // 
            this.edtAVIFile.Location = new System.Drawing.Point(104, 16);
            this.edtAVIFile.Name = "edtAVIFile";
            this.edtAVIFile.Size = new System.Drawing.Size(248, 20);
            this.edtAVIFile.TabIndex = 1;
            this.edtAVIFile.Text = "test.avi";
            // 
            // cmdAVIFileChooser
            // 
            this.cmdAVIFileChooser.Location = new System.Drawing.Point(360, 16);
            this.cmdAVIFileChooser.Name = "cmdAVIFileChooser";
            this.cmdAVIFileChooser.Size = new System.Drawing.Size(64, 24);
            this.cmdAVIFileChooser.TabIndex = 2;
            this.cmdAVIFileChooser.Text = "Browse";
            this.cmdAVIFileChooser.Click += new System.EventHandler(this.cmdAVIFileChooser_Click);
            // 
            // GroupBox2
            // 
            this.GroupBox2.Controls.Add(this.Label3);
            this.GroupBox2.Controls.Add(this.edtConfigFile);
            this.GroupBox2.Controls.Add(this.cmdCodecDataFileChooser);
            this.GroupBox2.Controls.Add(this.cmdSaveCodecData);
            this.GroupBox2.Controls.Add(this.cmdLoadCodecData);
            this.GroupBox2.Controls.Add(this.Label2);
            this.GroupBox2.Controls.Add(this.cmdShowPropPage);
            this.GroupBox2.Controls.Add(this.CodecBox);
            this.GroupBox2.Location = new System.Drawing.Point(12, 12);
            this.GroupBox2.Name = "GroupBox2";
            this.GroupBox2.Size = new System.Drawing.Size(456, 112);
            this.GroupBox2.TabIndex = 19;
            this.GroupBox2.TabStop = false;
            this.GroupBox2.Text = "Codec Data";
            // 
            // Label3
            // 
            this.Label3.Location = new System.Drawing.Point(8, 19);
            this.Label3.Name = "Label3";
            this.Label3.Size = new System.Drawing.Size(64, 16);
            this.Label3.TabIndex = 12;
            this.Label3.Text = "Codec:";
            // 
            // edtConfigFile
            // 
            this.edtConfigFile.Location = new System.Drawing.Point(104, 48);
            this.edtConfigFile.Name = "edtConfigFile";
            this.edtConfigFile.Size = new System.Drawing.Size(248, 20);
            this.edtConfigFile.TabIndex = 10;
            this.edtConfigFile.Text = "test.cod";
            // 
            // cmdCodecDataFileChooser
            // 
            this.cmdCodecDataFileChooser.Location = new System.Drawing.Point(360, 48);
            this.cmdCodecDataFileChooser.Name = "cmdCodecDataFileChooser";
            this.cmdCodecDataFileChooser.Size = new System.Drawing.Size(64, 24);
            this.cmdCodecDataFileChooser.TabIndex = 11;
            this.cmdCodecDataFileChooser.Text = "Browse";
            this.cmdCodecDataFileChooser.Click += new System.EventHandler(this.cmdCodecDataFileChooser_Click);
            // 
            // cmdSaveCodecData
            // 
            this.cmdSaveCodecData.Location = new System.Drawing.Point(64, 80);
            this.cmdSaveCodecData.Name = "cmdSaveCodecData";
            this.cmdSaveCodecData.Size = new System.Drawing.Size(128, 24);
            this.cmdSaveCodecData.TabIndex = 3;
            this.cmdSaveCodecData.Text = "Save Codec Data";
            this.cmdSaveCodecData.Click += new System.EventHandler(this.cmdSaveCodecData_Click);
            // 
            // cmdLoadCodecData
            // 
            this.cmdLoadCodecData.Location = new System.Drawing.Point(232, 80);
            this.cmdLoadCodecData.Name = "cmdLoadCodecData";
            this.cmdLoadCodecData.Size = new System.Drawing.Size(128, 24);
            this.cmdLoadCodecData.TabIndex = 4;
            this.cmdLoadCodecData.Text = "Load Codec Data";
            this.cmdLoadCodecData.Click += new System.EventHandler(this.cmdLoadCodecData_Click);
            // 
            // Label2
            // 
            this.Label2.Location = new System.Drawing.Point(8, 46);
            this.Label2.Name = "Label2";
            this.Label2.Size = new System.Drawing.Size(96, 26);
            this.Label2.TabIndex = 4;
            this.Label2.Text = "Codec Configuration file:";
            // 
            // cmdShowPropPage
            // 
            this.cmdShowPropPage.Location = new System.Drawing.Point(360, 16);
            this.cmdShowPropPage.Name = "cmdShowPropPage";
            this.cmdShowPropPage.Size = new System.Drawing.Size(64, 24);
            this.cmdShowPropPage.TabIndex = 5;
            this.cmdShowPropPage.Text = "Properties";
            this.cmdShowPropPage.Click += new System.EventHandler(this.cmdShowPropPage_Click);
            // 
            // CodecBox
            // 
            this.CodecBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.CodecBox.Location = new System.Drawing.Point(104, 16);
            this.CodecBox.Name = "CodecBox";
            this.CodecBox.Size = new System.Drawing.Size(248, 21);
            this.CodecBox.TabIndex = 0;
            this.CodecBox.SelectedIndexChanged += new System.EventHandler(this.CodecBox_Changed);
            // 
            // AviDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(479, 252);
            this.Controls.Add(this.chkCapturePause);
            this.Controls.Add(this.cmdStopButton);
            this.Controls.Add(this.cmdStartButton);
            this.Controls.Add(this.cmdCloseButton);
            this.Controls.Add(this.GroupBox1);
            this.Controls.Add(this.GroupBox2);
            this.MaximizeBox = false;
            this.MaximumSize = new System.Drawing.Size(487, 286);
            this.MinimizeBox = false;
            this.MinimumSize = new System.Drawing.Size(487, 286);
            this.Name = "AviDialog";
            this.Text = "Capture Avi File";
            this.Load += new System.EventHandler(this.AviDialog_Load);
            this.GroupBox1.ResumeLayout(false);
            this.GroupBox1.PerformLayout();
            this.GroupBox2.ResumeLayout(false);
            this.GroupBox2.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        internal System.Windows.Forms.CheckBox chkCapturePause;
        internal System.Windows.Forms.Button cmdStopButton;
        internal System.Windows.Forms.Button cmdStartButton;
        internal System.Windows.Forms.Button cmdCloseButton;
        internal System.Windows.Forms.GroupBox GroupBox1;
        internal System.Windows.Forms.Label Label1;
        internal System.Windows.Forms.TextBox edtAVIFile;
        internal System.Windows.Forms.Button cmdAVIFileChooser;
        internal System.Windows.Forms.GroupBox GroupBox2;
        internal System.Windows.Forms.Label Label3;
        internal System.Windows.Forms.TextBox edtConfigFile;
        internal System.Windows.Forms.Button cmdCodecDataFileChooser;
        internal System.Windows.Forms.Button cmdSaveCodecData;
        internal System.Windows.Forms.Button cmdLoadCodecData;
        internal System.Windows.Forms.Label Label2;
        internal System.Windows.Forms.Button cmdShowPropPage;
        internal System.Windows.Forms.ComboBox CodecBox;
    }
}