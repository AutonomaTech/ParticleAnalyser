namespace VCD_Simple_Property
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
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.BrightnessTrackBar = new System.Windows.Forms.TrackBar();
            this.WhiteBalRedTrackBar = new System.Windows.Forms.TrackBar();
            this.BrightnessValueLabel = new System.Windows.Forms.Label();
            this.WhiteBalRedLabel = new System.Windows.Forms.Label();
            this.BrightnessAutoCheckBox = new System.Windows.Forms.CheckBox();
            this.WhitebalanceCheckBox = new System.Windows.Forms.CheckBox();
            this.WhiteBalBlueTrackBar = new System.Windows.Forms.TrackBar();
            this.label3 = new System.Windows.Forms.Label();
            this.WhiteBalBlueLabel = new System.Windows.Forms.Label();
            this.WhitebalanceOnePushButton = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.BrightnessTrackBar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.WhiteBalRedTrackBar)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.WhiteBalBlueTrackBar)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(7, 7);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(536, 312);
            this.icImagingControl1.TabIndex = 0;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(15, 343);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(56, 13);
            this.label1.TabIndex = 1;
            this.label1.Text = "Brightness";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(15, 400);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(96, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Whitebalance Red";
            // 
            // BrightnessTrackBar
            // 
            this.BrightnessTrackBar.Location = new System.Drawing.Point(140, 334);
            this.BrightnessTrackBar.Name = "BrightnessTrackBar";
            this.BrightnessTrackBar.Size = new System.Drawing.Size(259, 45);
            this.BrightnessTrackBar.TabIndex = 3;
            this.BrightnessTrackBar.Scroll += new System.EventHandler(this.BrightnessTrackBar_Scroll);
            // 
            // WhiteBalRedTrackBar
            // 
            this.WhiteBalRedTrackBar.Location = new System.Drawing.Point(140, 385);
            this.WhiteBalRedTrackBar.Name = "WhiteBalRedTrackBar";
            this.WhiteBalRedTrackBar.Size = new System.Drawing.Size(259, 45);
            this.WhiteBalRedTrackBar.TabIndex = 4;
            this.WhiteBalRedTrackBar.Scroll += new System.EventHandler(this.WhiteBalRedTrackBar_Scroll);
            // 
            // BrightnessValueLabel
            // 
            this.BrightnessValueLabel.AutoSize = true;
            this.BrightnessValueLabel.Location = new System.Drawing.Point(418, 343);
            this.BrightnessValueLabel.Name = "BrightnessValueLabel";
            this.BrightnessValueLabel.Size = new System.Drawing.Size(13, 13);
            this.BrightnessValueLabel.TabIndex = 5;
            this.BrightnessValueLabel.Text = "0";
            // 
            // WhiteBalRedLabel
            // 
            this.WhiteBalRedLabel.AutoSize = true;
            this.WhiteBalRedLabel.Location = new System.Drawing.Point(418, 400);
            this.WhiteBalRedLabel.Name = "WhiteBalRedLabel";
            this.WhiteBalRedLabel.Size = new System.Drawing.Size(13, 13);
            this.WhiteBalRedLabel.TabIndex = 6;
            this.WhiteBalRedLabel.Text = "0";
            // 
            // BrightnessAutoCheckBox
            // 
            this.BrightnessAutoCheckBox.AutoSize = true;
            this.BrightnessAutoCheckBox.Location = new System.Drawing.Point(466, 339);
            this.BrightnessAutoCheckBox.Name = "BrightnessAutoCheckBox";
            this.BrightnessAutoCheckBox.Size = new System.Drawing.Size(48, 17);
            this.BrightnessAutoCheckBox.TabIndex = 7;
            this.BrightnessAutoCheckBox.Text = "Auto";
            this.BrightnessAutoCheckBox.UseVisualStyleBackColor = true;
            this.BrightnessAutoCheckBox.CheckedChanged += new System.EventHandler(this.BrightnessAutoCheckBox_CheckedChanged);
            // 
            // WhitebalanceCheckBox
            // 
            this.WhitebalanceCheckBox.AutoSize = true;
            this.WhitebalanceCheckBox.Location = new System.Drawing.Point(466, 397);
            this.WhitebalanceCheckBox.Name = "WhitebalanceCheckBox";
            this.WhitebalanceCheckBox.Size = new System.Drawing.Size(48, 17);
            this.WhitebalanceCheckBox.TabIndex = 8;
            this.WhitebalanceCheckBox.Text = "Auto";
            this.WhitebalanceCheckBox.UseVisualStyleBackColor = true;
            this.WhitebalanceCheckBox.CheckedChanged += new System.EventHandler(this.WhitebalanceCheckBox_CheckedChanged);
            // 
            // WhiteBalBlueTrackBar
            // 
            this.WhiteBalBlueTrackBar.Location = new System.Drawing.Point(140, 436);
            this.WhiteBalBlueTrackBar.Name = "WhiteBalBlueTrackBar";
            this.WhiteBalBlueTrackBar.Size = new System.Drawing.Size(259, 45);
            this.WhiteBalBlueTrackBar.TabIndex = 9;
            this.WhiteBalBlueTrackBar.Scroll += new System.EventHandler(this.WhiteBalBlueTrackBar_Scroll);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(15, 445);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(97, 13);
            this.label3.TabIndex = 10;
            this.label3.Text = "Whitebalance Blue";
            // 
            // WhiteBalBlueLabel
            // 
            this.WhiteBalBlueLabel.AutoSize = true;
            this.WhiteBalBlueLabel.Location = new System.Drawing.Point(418, 445);
            this.WhiteBalBlueLabel.Name = "WhiteBalBlueLabel";
            this.WhiteBalBlueLabel.Size = new System.Drawing.Size(13, 13);
            this.WhiteBalBlueLabel.TabIndex = 11;
            this.WhiteBalBlueLabel.Text = "0";
            // 
            // WhitebalanceOnePushButton
            // 
            this.WhitebalanceOnePushButton.Location = new System.Drawing.Point(466, 420);
            this.WhitebalanceOnePushButton.Name = "WhitebalanceOnePushButton";
            this.WhitebalanceOnePushButton.Size = new System.Drawing.Size(68, 24);
            this.WhitebalanceOnePushButton.TabIndex = 12;
            this.WhitebalanceOnePushButton.Text = "One Push";
            this.WhitebalanceOnePushButton.UseVisualStyleBackColor = true;
            this.WhitebalanceOnePushButton.Click += new System.EventHandler(this.WhitebalanceOnePushButton_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(552, 499);
            this.Controls.Add(this.WhitebalanceOnePushButton);
            this.Controls.Add(this.WhiteBalBlueLabel);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.WhiteBalBlueTrackBar);
            this.Controls.Add(this.WhitebalanceCheckBox);
            this.Controls.Add(this.BrightnessAutoCheckBox);
            this.Controls.Add(this.WhiteBalRedLabel);
            this.Controls.Add(this.BrightnessValueLabel);
            this.Controls.Add(this.WhiteBalRedTrackBar);
            this.Controls.Add(this.BrightnessTrackBar);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.icImagingControl1);
            this.Name = "Form1";
            this.Text = "VCD Simple Property";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.BrightnessTrackBar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.WhiteBalRedTrackBar)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.WhiteBalBlueTrackBar)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TrackBar BrightnessTrackBar;
        private System.Windows.Forms.TrackBar WhiteBalRedTrackBar;
        private System.Windows.Forms.Label BrightnessValueLabel;
        private System.Windows.Forms.Label WhiteBalRedLabel;
        private System.Windows.Forms.CheckBox BrightnessAutoCheckBox;
        private System.Windows.Forms.CheckBox WhitebalanceCheckBox;
        private System.Windows.Forms.TrackBar WhiteBalBlueTrackBar;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label WhiteBalBlueLabel;
        private System.Windows.Forms.Button WhitebalanceOnePushButton;
    }
}

