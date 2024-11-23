namespace MakingDeviceSettings
{
	partial class frmDeviceSettings
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose( bool disposing )
		{
			if( disposing && ( components != null ) )
			{
				components.Dispose();
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.cboDevice = new System.Windows.Forms.ComboBox();
			this.cboVideoNorm = new System.Windows.Forms.ComboBox();
			this.cboVideoFormat = new System.Windows.Forms.ComboBox();
			this.cboFrameRate = new System.Windows.Forms.ComboBox();
			this.label3 = new System.Windows.Forms.Label();
			this.label4 = new System.Windows.Forms.Label();
			this.cboInputChannel = new System.Windows.Forms.ComboBox();
			this.label5 = new System.Windows.Forms.Label();
			this.txtSerial = new System.Windows.Forms.TextBox();
			this.label6 = new System.Windows.Forms.Label();
			this.chkFlipV = new System.Windows.Forms.CheckBox();
			this.chkFlipH = new System.Windows.Forms.CheckBox();
			this.cmdCancel = new System.Windows.Forms.Button();
			this.cmdOK = new System.Windows.Forms.Button();
			this.lblErrorMessage = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// label1
			// 
			this.label1.AutoSize = true;
			this.label1.Location = new System.Drawing.Point( 12, 12 );
			this.label1.Name = "label1";
			this.label1.Size = new System.Drawing.Size( 41, 13 );
			this.label1.TabIndex = 0;
			this.label1.Text = "Device";
			// 
			// label2
			// 
			this.label2.AutoSize = true;
			this.label2.Location = new System.Drawing.Point( 12, 39 );
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size( 62, 13 );
			this.label2.TabIndex = 1;
			this.label2.Text = "Video Norm";
			// 
			// cboDevice
			// 
			this.cboDevice.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cboDevice.FormattingEnabled = true;
			this.cboDevice.Location = new System.Drawing.Point( 106, 9 );
			this.cboDevice.Name = "cboDevice";
			this.cboDevice.Size = new System.Drawing.Size( 185, 21 );
			this.cboDevice.TabIndex = 2;
			this.cboDevice.SelectedIndexChanged += new System.EventHandler( this.cboDevice_SelectedIndexChanged );
			// 
			// cboVideoNorm
			// 
			this.cboVideoNorm.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cboVideoNorm.FormattingEnabled = true;
			this.cboVideoNorm.Location = new System.Drawing.Point( 106, 36 );
			this.cboVideoNorm.Name = "cboVideoNorm";
			this.cboVideoNorm.Size = new System.Drawing.Size( 185, 21 );
			this.cboVideoNorm.TabIndex = 3;
			this.cboVideoNorm.SelectedIndexChanged += new System.EventHandler( this.cboVideoNorm_SelectedIndexChanged );
			// 
			// cboVideoFormat
			// 
			this.cboVideoFormat.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cboVideoFormat.FormattingEnabled = true;
			this.cboVideoFormat.Location = new System.Drawing.Point( 106, 63 );
			this.cboVideoFormat.Name = "cboVideoFormat";
			this.cboVideoFormat.Size = new System.Drawing.Size( 185, 21 );
			this.cboVideoFormat.TabIndex = 4;
			this.cboVideoFormat.SelectedIndexChanged += new System.EventHandler( this.cboVideoFormat_SelectedIndexChanged );
			// 
			// cboFrameRate
			// 
			this.cboFrameRate.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cboFrameRate.FormattingEnabled = true;
			this.cboFrameRate.Location = new System.Drawing.Point( 106, 90 );
			this.cboFrameRate.Name = "cboFrameRate";
			this.cboFrameRate.Size = new System.Drawing.Size( 185, 21 );
			this.cboFrameRate.TabIndex = 5;
			this.cboFrameRate.SelectedIndexChanged += new System.EventHandler( this.cboFrameRate_SelectedIndexChanged );
			// 
			// label3
			// 
			this.label3.AutoSize = true;
			this.label3.Location = new System.Drawing.Point( 12, 66 );
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size( 69, 13 );
			this.label3.TabIndex = 6;
			this.label3.Text = "Video Format";
			// 
			// label4
			// 
			this.label4.AutoSize = true;
			this.label4.Location = new System.Drawing.Point( 12, 93 );
			this.label4.Name = "label4";
			this.label4.Size = new System.Drawing.Size( 62, 13 );
			this.label4.TabIndex = 7;
			this.label4.Text = "Frame Rate";
			// 
			// cboInputChannel
			// 
			this.cboInputChannel.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
			this.cboInputChannel.FormattingEnabled = true;
			this.cboInputChannel.Location = new System.Drawing.Point( 106, 117 );
			this.cboInputChannel.Name = "cboInputChannel";
			this.cboInputChannel.Size = new System.Drawing.Size( 185, 21 );
			this.cboInputChannel.TabIndex = 8;
			this.cboInputChannel.SelectedIndexChanged += new System.EventHandler( this.cboInputChannel_SelectedIndexChanged );
			// 
			// label5
			// 
			this.label5.AutoSize = true;
			this.label5.Location = new System.Drawing.Point( 12, 120 );
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size( 73, 13 );
			this.label5.TabIndex = 9;
			this.label5.Text = "Input Channel";
			// 
			// txtSerial
			// 
			this.txtSerial.Location = new System.Drawing.Point( 106, 144 );
			this.txtSerial.Name = "txtSerial";
			this.txtSerial.ReadOnly = true;
			this.txtSerial.Size = new System.Drawing.Size( 185, 20 );
			this.txtSerial.TabIndex = 10;
			// 
			// label6
			// 
			this.label6.AutoSize = true;
			this.label6.Location = new System.Drawing.Point( 12, 147 );
			this.label6.Name = "label6";
			this.label6.Size = new System.Drawing.Size( 73, 13 );
			this.label6.TabIndex = 11;
			this.label6.Text = "Serial Number";
			// 
			// chkFlipV
			// 
			this.chkFlipV.AutoSize = true;
			this.chkFlipV.Location = new System.Drawing.Point( 106, 170 );
			this.chkFlipV.Name = "chkFlipV";
			this.chkFlipV.Size = new System.Drawing.Size( 80, 17 );
			this.chkFlipV.TabIndex = 12;
			this.chkFlipV.Text = "Flip Vertical";
			this.chkFlipV.UseVisualStyleBackColor = true;
			this.chkFlipV.CheckedChanged += new System.EventHandler( this.chkFlipV_CheckedChanged );
			// 
			// chkFlipH
			// 
			this.chkFlipH.AutoSize = true;
			this.chkFlipH.Location = new System.Drawing.Point( 199, 170 );
			this.chkFlipH.Name = "chkFlipH";
			this.chkFlipH.Size = new System.Drawing.Size( 92, 17 );
			this.chkFlipH.TabIndex = 13;
			this.chkFlipH.Text = "Flip Horizontal";
			this.chkFlipH.UseVisualStyleBackColor = true;
			this.chkFlipH.CheckedChanged += new System.EventHandler( this.chkFlipH_CheckedChanged );
			// 
			// cmdCancel
			// 
			this.cmdCancel.DialogResult = System.Windows.Forms.DialogResult.Cancel;
			this.cmdCancel.Location = new System.Drawing.Point( 216, 195 );
			this.cmdCancel.Name = "cmdCancel";
			this.cmdCancel.Size = new System.Drawing.Size( 75, 23 );
			this.cmdCancel.TabIndex = 14;
			this.cmdCancel.Text = "Cancel";
			this.cmdCancel.UseVisualStyleBackColor = true;
			this.cmdCancel.Click += new System.EventHandler( this.cmdCancel_Click );
			// 
			// cmdOK
			// 
			this.cmdOK.DialogResult = System.Windows.Forms.DialogResult.OK;
			this.cmdOK.Location = new System.Drawing.Point( 135, 195 );
			this.cmdOK.Name = "cmdOK";
			this.cmdOK.Size = new System.Drawing.Size( 75, 23 );
			this.cmdOK.TabIndex = 15;
			this.cmdOK.Text = "OK";
			this.cmdOK.UseVisualStyleBackColor = true;
			this.cmdOK.Click += new System.EventHandler( this.cmdOK_Click );
			// 
			// lblErrorMessage
			// 
			this.lblErrorMessage.AutoSize = true;
			this.lblErrorMessage.Location = new System.Drawing.Point( 16, 179 );
			this.lblErrorMessage.Name = "lblErrorMessage";
			this.lblErrorMessage.Size = new System.Drawing.Size( 70, 13 );
			this.lblErrorMessage.TabIndex = 16;
			this.lblErrorMessage.Text = "<Error Label>";
			this.lblErrorMessage.Visible = false;
			// 
			// frmDeviceSettings
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF( 6F, 13F );
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size( 303, 230 );
			this.Controls.Add( this.lblErrorMessage );
			this.Controls.Add( this.cmdOK );
			this.Controls.Add( this.cmdCancel );
			this.Controls.Add( this.chkFlipH );
			this.Controls.Add( this.chkFlipV );
			this.Controls.Add( this.label6 );
			this.Controls.Add( this.txtSerial );
			this.Controls.Add( this.label5 );
			this.Controls.Add( this.cboInputChannel );
			this.Controls.Add( this.label4 );
			this.Controls.Add( this.label3 );
			this.Controls.Add( this.cboFrameRate );
			this.Controls.Add( this.cboVideoFormat );
			this.Controls.Add( this.cboVideoNorm );
			this.Controls.Add( this.cboDevice );
			this.Controls.Add( this.label2 );
			this.Controls.Add( this.label1 );
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.Name = "frmDeviceSettings";
			this.Text = "Custom Device Settings Dialog";
			this.Load += new System.EventHandler( this.frmDeviceSettings_Load );
			this.ResumeLayout( false );
			this.PerformLayout();

		}

		#endregion

		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.ComboBox cboDevice;
		private System.Windows.Forms.ComboBox cboVideoNorm;
		private System.Windows.Forms.ComboBox cboVideoFormat;
		private System.Windows.Forms.ComboBox cboFrameRate;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.ComboBox cboInputChannel;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.TextBox txtSerial;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.CheckBox chkFlipV;
		private System.Windows.Forms.CheckBox chkFlipH;
		private System.Windows.Forms.Button cmdCancel;
		private System.Windows.Forms.Button cmdOK;
		private System.Windows.Forms.Label lblErrorMessage;
	}
}