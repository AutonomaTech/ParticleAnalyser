using System;
using System.Collections;
using System.Data;
using System.Drawing;
using System.Diagnostics;
using System.Windows.Forms;

namespace demoapp
{
    internal class DeviceSettings
    // Constants for Settings dialog box
    {
        const string REG_APP_NAME = "TIS CS .NET DemoApp";
        const string REG_SECTION = "Settings";
        const string REG_KEY_VIDEO_SOURCE = "VideoSource";
        const string REG_KEY_VIDEO_NORM = "VideoNorm";
        const string REG_KEY_INPUT_CHANNEL = "InputChannel";
        const string REG_KEY_VIDEO_FORMAT = "VideoFormat";
        const string REG_KEY_WINDOW_SIZE = "Adjust Window Size";
        const string REG_KEY_FRAME_RATE = "Frame Rate";
        const string REG_KEY_CODEC = "Codec";
        const string REG_KEY_CODECDATA = "Codec Data";
        const string REG_KEY_AVIFILE = "AVI File";

        //
        // CurrentProcessName
        //
        // Return the name of the current process. This fuction is used in the
        // GetSetting and SaveSetting function as section parameter
        // This makes this module useable for other applications too.
        //
        internal static object CurrentProcessName()
        {
            return System.Diagnostics.Process.GetCurrentProcess().ProcessName;
        }
        //
        // OpenSelectedDevice
        //
        // Open the device that has be stored in the registry last time, the
        // device selection dialog has been used. Call this sub in the beginning
        // of an application to restore the last used video capture device.
        //
        internal static bool OpenSelectedDevice( TIS.Imaging.ICImagingControl icImagingControl1 )
        {
            string ActiveErrorHandler = "";

            bool OpenSelectedDevice_temp = false;

            try
            {
                //        On Error GoTo err_OpenSelectedDevice
                ActiveErrorHandler = "err_OpenSelectedDevice";

                icImagingControl1.Device = Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_VIDEO_SOURCE, "" );

                // Restore settings
                if( icImagingControl1.VideoNormAvailable )
                {
                    icImagingControl1.VideoNorm = Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_VIDEO_NORM, "" );
                }
                if( icImagingControl1.InputChannelAvailable )
                {
                    icImagingControl1.InputChannel = Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_INPUT_CHANNEL, "" );
                }
                icImagingControl1.VideoFormat = Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_VIDEO_FORMAT, "" );

                if( icImagingControl1.DeviceFrameRateAvailable )
                {
                    icImagingControl1.DeviceFrameRate = float.Parse( Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_FRAME_RATE, "" ) );
                }
                // No error occured, and device is present. Indicate success
                OpenSelectedDevice_temp = true;

                return OpenSelectedDevice_temp;
            }
            catch
            {
                if( ActiveErrorHandler == "err_OpenSelectedDevice" )
                {
                    return OpenSelectedDevice_temp;
                }
            }

            return OpenSelectedDevice_temp;
        }

        //
        // SaveSelectedDevice
        //
        // Saves the device settings from icImagingControl1 to the registry
        //
        internal static void SaveSelectedDevice( TIS.Imaging.ICImagingControl icImagingControl1 )
        {
            string ActiveErrorHandler = "";
            try
            {
                //        On Error GoTo err_SaveSelectedDevice
                ActiveErrorHandler = "err_SaveSelectedDevice";

                if( icImagingControl1.DeviceValid )
                {
                    Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_VIDEO_SOURCE, icImagingControl1.Device );
                    if( icImagingControl1.InputChannelAvailable )
                    {
                        Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_INPUT_CHANNEL, icImagingControl1.InputChannel );
                    }
                    if( icImagingControl1.VideoNormAvailable )
                    {
                        Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_VIDEO_NORM, icImagingControl1.VideoNorm );
                    }

                    Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_VIDEO_FORMAT, icImagingControl1.VideoFormat );

                    if( icImagingControl1.DeviceFrameRateAvailable )
                    {
                        Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_FRAME_RATE, icImagingControl1.DeviceFrameRate.ToString() );
                    }

                }
            }

            catch
            {
                if( ActiveErrorHandler == "err_SaveSelectedDevice" )
                {
                }
            }
        }

        //
        // GetCodecFromRegisty
        //
        // Retrieve the last used codec from the registry.
        //
        internal static string GetCodecFromRegisty()
        {
            return Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_CODEC, "" );
        }

        //
        // SaveCodecInRegisty
        //
        // Save the last used codec  file in the registry.
        //
        internal static void SaveCodecInRegisty( string Codec )
        {
            Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_CODEC, Codec );
        }

        //
        // GetCodecFileFromRegisty
        //
        // Retrieve the last used codec configuration file from the registry.
        //
        internal static string GetCodecFileFromRegisty()
        {
            return Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_CODECDATA, "new.cod" );
        }

        //
        // SaveCodecFileInRegisty
        //
        // Save the last used codec configuration file to the registry.
        //
        internal static void SaveCodecFileInRegisty( string FileName )
        {
            Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_CODECDATA, FileName );
        }

        //
        // GetAVIFileFromRegisty
        //
        // Retrieve the last used AVI file from the registry.
        //
        internal static string GetAVIFileFromRegisty()
        {
            return Microsoft.VisualBasic.Interaction.GetSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_AVIFILE, "new.avi" );
        }

        //
        // SaveAVIFileInRegisty
        //
        // Retrieve the last used AVI file from the registry
        //
        internal static void SaveAVIFileInRegisty( string FileName )
        {
            Microsoft.VisualBasic.Interaction.SaveSetting( DeviceSettings.CurrentProcessName().ToString(), REG_SECTION, REG_KEY_AVIFILE, FileName );
        }

    }
}
