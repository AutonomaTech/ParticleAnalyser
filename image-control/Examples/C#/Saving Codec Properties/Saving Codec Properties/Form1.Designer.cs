namespace Saving_Codec_Properties
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
            this.cboVideoCodec = new System.Windows.Forms.ComboBox();
            this.cmdSaveData = new System.Windows.Forms.Button();
            this.cmdLoadData = new System.Windows.Forms.Button();
            this.cmdShowPropertyPage = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).BeginInit();
            this.SuspendLayout();
            // 
            // icImagingControl1
            // 
            this.icImagingControl1.BackColor = System.Drawing.Color.White;
            this.icImagingControl1.LiveDisplayPosition = new System.Drawing.Point(0, 0);
            this.icImagingControl1.Location = new System.Drawing.Point(8, 9);
            this.icImagingControl1.Name = "icImagingControl1";
            this.icImagingControl1.Size = new System.Drawing.Size(344, 160);
            this.icImagingControl1.TabIndex = 0;
            // 
            // cboVideoCodec
            // 
            this.cboVideoCodec.FormattingEnabled = true;
            this.cboVideoCodec.Location = new System.Drawing.Point(8, 175);
            this.cboVideoCodec.Name = "cboVideoCodec";
            this.cboVideoCodec.Size = new System.Drawing.Size(344, 21);
            this.cboVideoCodec.TabIndex = 1;
            this.cboVideoCodec.SelectedIndexChanged += new System.EventHandler(this.cboVideoCodec_SelectedIndexChanged);
            // 
            // cmdSaveData
            // 
            this.cmdSaveData.Location = new System.Drawing.Point(8, 202);
            this.cmdSaveData.Name = "cmdSaveData";
            this.cmdSaveData.Size = new System.Drawing.Size(75, 23);
            this.cmdSaveData.TabIndex = 2;
            this.cmdSaveData.Text = "Save Data";
            this.cmdSaveData.UseVisualStyleBackColor = true;
            this.cmdSaveData.Click += new System.EventHandler(this.cmdSaveData_Click);
            // 
            // cmdLoadData
            // 
            this.cmdLoadData.Location = new System.Drawing.Point(89, 202);
            this.cmdLoadData.Name = "cmdLoadData";
            this.cmdLoadData.Size = new System.Drawing.Size(75, 23);
            this.cmdLoadData.TabIndex = 3;
            this.cmdLoadData.Text = "Load Data";
            this.cmdLoadData.UseVisualStyleBackColor = true;
            this.cmdLoadData.Click += new System.EventHandler(this.cmdLoadData_Click);
            // 
            // cmdShowPropertyPage
            // 
            this.cmdShowPropertyPage.Location = new System.Drawing.Point(240, 202);
            this.cmdShowPropertyPage.Name = "cmdShowPropertyPage";
            this.cmdShowPropertyPage.Size = new System.Drawing.Size(112, 23);
            this.cmdShowPropertyPage.TabIndex = 4;
            this.cmdShowPropertyPage.Text = "Show Property Page";
            this.cmdShowPropertyPage.UseVisualStyleBackColor = true;
            this.cmdShowPropertyPage.Click += new System.EventHandler(this.cmdShowPropertyPage_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(360, 237);
            this.Controls.Add(this.cmdShowPropertyPage);
            this.Controls.Add(this.cmdLoadData);
            this.Controls.Add(this.cmdSaveData);
            this.Controls.Add(this.cboVideoCodec);
            this.Controls.Add(this.icImagingControl1);
            this.Name = "Form1";
            this.Text = "Saving Codec Properties";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.icImagingControl1)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private TIS.Imaging.ICImagingControl icImagingControl1;
        private System.Windows.Forms.ComboBox cboVideoCodec;
        private System.Windows.Forms.Button cmdSaveData;
        private System.Windows.Forms.Button cmdLoadData;
        private System.Windows.Forms.Button cmdShowPropertyPage;
    }
}

