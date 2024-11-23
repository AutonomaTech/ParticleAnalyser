namespace demoapp
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.MainMenu1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveImageToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.writeAviToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.previewToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.playToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.stopToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.deviceSettingsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.deviceToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.imageToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.inputChannelsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.externalTriggerToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStrip1 = new System.Windows.Forms.ToolStrip();
            this.ToolbarPlayButton = new System.Windows.Forms.ToolStripButton();
            this.ToolBarStopButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator2 = new System.Windows.Forms.ToolStripSeparator();
            this.ToolBarSnapButton = new System.Windows.Forms.ToolStripButton();
            this.ToolBarAVIButton = new System.Windows.Forms.ToolStripButton();
            this.toolStripSeparator3 = new System.Windows.Forms.ToolStripSeparator();
            this.ToolBarDeviceButton = new System.Windows.Forms.ToolStripButton();
            this.ToolBarPropertiesButton = new System.Windows.Forms.ToolStripButton();
            this.ToolBarInputChannel = new System.Windows.Forms.ToolStripDropDownButton();
            this.toolStripSeparator4 = new System.Windows.Forms.ToolStripSeparator();
            this.ToolBarTriggerButton = new System.Windows.Forms.ToolStripButton();
            this.icImagingControl1 = new TIS.Imaging.ICImagingControl();
            this.MainMenu1.SuspendLayout();
            this.toolStrip1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // MainMenu1
            // 
            this.MainMenu1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.previewToolStripMenuItem,
            this.deviceSettingsToolStripMenuItem});
            this.MainMenu1.Location = new System.Drawing.Point(0, 0);
            this.MainMenu1.Name = "MainMenu1";
            this.MainMenu1.Size = new System.Drawing.Size(360, 24);
            this.MainMenu1.TabIndex = 0;
            this.MainMenu1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.saveImageToolStripMenuItem,
            this.writeAviToolStripMenuItem,
            this.toolStripSeparator1,
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(35, 20);
            this.fileToolStripMenuItem.Text = "File";
            // 
            // saveImageToolStripMenuItem
            // 
            this.saveImageToolStripMenuItem.Name = "saveImageToolStripMenuItem";
            this.saveImageToolStripMenuItem.Size = new System.Drawing.Size(131, 22);
            this.saveImageToolStripMenuItem.Text = "Save Image";
            this.saveImageToolStripMenuItem.Click += new System.EventHandler(this.saveImageToolStripMenuItem_Click);
            // 
            // writeAviToolStripMenuItem
            // 
            this.writeAviToolStripMenuItem.Name = "writeAviToolStripMenuItem";
            this.writeAviToolStripMenuItem.Size = new System.Drawing.Size(131, 22);
            this.writeAviToolStripMenuItem.Text = "Write Avi...";
            this.writeAviToolStripMenuItem.Click += new System.EventHandler(this.writeAviToolStripMenuItem_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(128, 6);
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(131, 22);
            this.exitToolStripMenuItem.Text = "Exit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // previewToolStripMenuItem
            // 
            this.previewToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.playToolStripMenuItem,
            this.stopToolStripMenuItem});
            this.previewToolStripMenuItem.Name = "previewToolStripMenuItem";
            this.previewToolStripMenuItem.Size = new System.Drawing.Size(57, 20);
            this.previewToolStripMenuItem.Text = "Preview";
            // 
            // playToolStripMenuItem
            // 
            this.playToolStripMenuItem.Name = "playToolStripMenuItem";
            this.playToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.playToolStripMenuItem.Text = "Play";
            this.playToolStripMenuItem.Click += new System.EventHandler(this.playToolStripMenuItem_Click);
            // 
            // stopToolStripMenuItem
            // 
            this.stopToolStripMenuItem.Name = "stopToolStripMenuItem";
            this.stopToolStripMenuItem.Size = new System.Drawing.Size(152, 22);
            this.stopToolStripMenuItem.Text = "Stop";
            this.stopToolStripMenuItem.Click += new System.EventHandler(this.stopToolStripMenuItem_Click);
            // 
            // deviceSettingsToolStripMenuItem
            // 
            this.deviceSettingsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.deviceToolStripMenuItem,
            this.imageToolStripMenuItem,
            this.inputChannelsToolStripMenuItem,
            this.externalTriggerToolStripMenuItem});
            this.deviceSettingsToolStripMenuItem.Name = "deviceSettingsToolStripMenuItem";
            this.deviceSettingsToolStripMenuItem.Size = new System.Drawing.Size(93, 20);
            this.deviceSettingsToolStripMenuItem.Text = "Device Settings";
            // 
            // deviceToolStripMenuItem
            // 
            this.deviceToolStripMenuItem.Name = "deviceToolStripMenuItem";
            this.deviceToolStripMenuItem.Size = new System.Drawing.Size(151, 22);
            this.deviceToolStripMenuItem.Text = "Device...";
            this.deviceToolStripMenuItem.Click += new System.EventHandler(this.deviceToolStripMenuItem_Click);
            // 
            // imageToolStripMenuItem
            // 
            this.imageToolStripMenuItem.Name = "imageToolStripMenuItem";
            this.imageToolStripMenuItem.Size = new System.Drawing.Size(151, 22);
            this.imageToolStripMenuItem.Text = "Image...";
            this.imageToolStripMenuItem.Click += new System.EventHandler(this.imageToolStripMenuItem_Click);
            // 
            // inputChannelsToolStripMenuItem
            // 
            this.inputChannelsToolStripMenuItem.Name = "inputChannelsToolStripMenuItem";
            this.inputChannelsToolStripMenuItem.Size = new System.Drawing.Size(151, 22);
            this.inputChannelsToolStripMenuItem.Text = "Input Channels";
            // 
            // externalTriggerToolStripMenuItem
            // 
            this.externalTriggerToolStripMenuItem.Name = "externalTriggerToolStripMenuItem";
            this.externalTriggerToolStripMenuItem.Size = new System.Drawing.Size(151, 22);
            this.externalTriggerToolStripMenuItem.Text = "External Trigger";
            this.externalTriggerToolStripMenuItem.Click += new System.EventHandler(this.externalTriggerToolStripMenuItem_Click);
            // 
            // toolStrip1
            // 
            this.toolStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.ToolbarPlayButton,
            this.ToolBarStopButton,
            this.toolStripSeparator2,
            this.ToolBarSnapButton,
            this.ToolBarAVIButton,
            this.toolStripSeparator3,
            this.ToolBarDeviceButton,
            this.ToolBarPropertiesButton,
            this.ToolBarInputChannel,
            this.toolStripSeparator4,
            this.ToolBarTriggerButton});
            this.toolStrip1.Location = new System.Drawing.Point(0, 24);
            this.toolStrip1.Name = "toolStrip1";
            this.toolStrip1.Size = new System.Drawing.Size(360, 25);
            this.toolStrip1.TabIndex = 1;
            this.toolStrip1.Text = "toolStrip1";
            this.toolStrip1.ItemClicked += new System.Windows.Forms.ToolStripItemClickedEventHandler(this.toolStrip1_ItemClicked);
            // 
            // ToolbarPlayButton
            // 
            this.ToolbarPlayButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolbarPlayButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolbarPlayButton.Image")));
            this.ToolbarPlayButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolbarPlayButton.Name = "ToolbarPlayButton";
            this.ToolbarPlayButton.Size = new System.Drawing.Size(23, 22);
			this.ToolbarPlayButton.Text = "Start the live video.";
            // 
            // ToolBarStopButton
            // 
            this.ToolBarStopButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarStopButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarStopButton.Image")));
            this.ToolBarStopButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarStopButton.Name = "ToolBarStopButton";
            this.ToolBarStopButton.Size = new System.Drawing.Size(23, 22);
			this.ToolBarStopButton.Text = "Stop the live video.";
            // 
            // toolStripSeparator2
            // 
            this.toolStripSeparator2.Name = "toolStripSeparator2";
            this.toolStripSeparator2.Size = new System.Drawing.Size(6, 25);
            // 
            // ToolBarSnapButton
            // 
            this.ToolBarSnapButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarSnapButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarSnapButton.Image")));
            this.ToolBarSnapButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarSnapButton.Name = "ToolBarSnapButton";
            this.ToolBarSnapButton.Size = new System.Drawing.Size(23, 22);
			this.ToolBarSnapButton.Text = "Snap an image and save it to disk.";
            // 
            // ToolBarAVIButton
            // 
            this.ToolBarAVIButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarAVIButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarAVIButton.Image")));
            this.ToolBarAVIButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarAVIButton.Name = "ToolBarAVIButton";
            this.ToolBarAVIButton.Size = new System.Drawing.Size(23, 22);
			this.ToolBarAVIButton.Text = "Open the AVI Capture dialog";
            // 
            // toolStripSeparator3
            // 
            this.toolStripSeparator3.Name = "toolStripSeparator3";
            this.toolStripSeparator3.Size = new System.Drawing.Size(6, 25);
            // 
            // ToolBarDeviceButton
            // 
            this.ToolBarDeviceButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarDeviceButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarDeviceButton.Image")));
            this.ToolBarDeviceButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarDeviceButton.Name = "ToolBarDeviceButton";
            this.ToolBarDeviceButton.Size = new System.Drawing.Size(23, 22);
			this.ToolBarDeviceButton.Text = "Select a video capture device, video format and frame rate.";
            // 
            // ToolBarPropertiesButton
            // 
            this.ToolBarPropertiesButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarPropertiesButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarPropertiesButton.Image")));
            this.ToolBarPropertiesButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarPropertiesButton.Name = "ToolBarPropertiesButton";
            this.ToolBarPropertiesButton.Size = new System.Drawing.Size(23, 22);
			this.ToolBarPropertiesButton.Text = "Show camera and video properties dialog.";
            // 
            // ToolBarInputChannel
            // 
            this.ToolBarInputChannel.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarInputChannel.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarInputChannel.Image")));
            this.ToolBarInputChannel.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarInputChannel.Name = "ToolBarInputChannel";
            this.ToolBarInputChannel.Size = new System.Drawing.Size(29, 22);
			this.ToolBarInputChannel.Text = "Select an input channel.";
            // 
            // toolStripSeparator4
            // 
            this.toolStripSeparator4.Name = "toolStripSeparator4";
            this.toolStripSeparator4.Size = new System.Drawing.Size(6, 25);
            // 
            // ToolBarTriggerButton
            // 
            this.ToolBarTriggerButton.CheckOnClick = true;
            this.ToolBarTriggerButton.DisplayStyle = System.Windows.Forms.ToolStripItemDisplayStyle.Image;
            this.ToolBarTriggerButton.Image = ((System.Drawing.Image)(resources.GetObject("ToolBarTriggerButton.Image")));
            this.ToolBarTriggerButton.ImageTransparentColor = System.Drawing.Color.Magenta;
            this.ToolBarTriggerButton.Name = "ToolBarTriggerButton";
            this.ToolBarTriggerButton.Size = new System.Drawing.Size(23, 22);
			this.ToolBarTriggerButton.Text = "Enable or disable the external trigger.";
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                        | System.Windows.Forms.AnchorStyles.Left)
                        | System.Windows.Forms.AnchorStyles.Right)));
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(0, 52);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(360, 282);
            this.icImagingControl1.TabIndex = 2;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(360, 334);
            this.Controls.Add(this.icImagingControl1);
            this.Controls.Add(this.toolStrip1);
            this.Controls.Add(this.MainMenu1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MainMenuStrip = this.MainMenu1;
            this.Name = "Form1";
            this.Text = "Demo Application C#";
            this.Resize += new System.EventHandler(this.Form1_Resize);
            this.Load += new System.EventHandler(this.Form1_Load);
            this.MainMenu1.ResumeLayout(false);
            this.MainMenu1.PerformLayout();
            this.toolStrip1.ResumeLayout(false);
            this.toolStrip1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip MainMenu1;
        private System.Windows.Forms.ToolStrip toolStrip1;
        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem saveImageToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem writeAviToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem previewToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem playToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem stopToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem deviceSettingsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem deviceToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem imageToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem inputChannelsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem externalTriggerToolStripMenuItem;
        private System.Windows.Forms.ToolStripButton ToolbarPlayButton;
        private System.Windows.Forms.ToolStripButton ToolBarStopButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator2;
        private System.Windows.Forms.ToolStripButton ToolBarSnapButton;
        private System.Windows.Forms.ToolStripButton ToolBarAVIButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator3;
        private System.Windows.Forms.ToolStripButton ToolBarDeviceButton;
        private System.Windows.Forms.ToolStripButton ToolBarPropertiesButton;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator4;
        private System.Windows.Forms.ToolStripButton ToolBarTriggerButton;
        private System.Windows.Forms.ToolStripDropDownButton ToolBarInputChannel;
    }
}

