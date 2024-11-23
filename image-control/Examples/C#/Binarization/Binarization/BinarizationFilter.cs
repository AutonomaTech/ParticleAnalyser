using System;
using System.Collections.Generic;
using System.Text;
using TIS.Imaging;

namespace Binarization
{
    /// <summary>
    /// This frame filter applies a binarization on the image data.
    /// If enabled, every gray value greater of equal to a specified threshold is changed to 
    /// the maximum gray value, every other gray value is changed to zero.
    /// 
    /// Allowed input types: Y800, RGB8
    /// 
    /// Output types: Y800, RGB8, the input type determines the output type.
    /// 
    ///	Parameters:
    ///		enable:
    ///				Boolean. Used to enable or disable binarization.
    ///				If binarization is disabled, the image data is not modified.
    ///		threshold:
    ///				Integer. Used to set the threshold for the binarization.
    /// </summary>
//<<classDef
    public class BinarizationFilter : FrameFilterImpl
    //>>
    {
        //<<membervars
        private bool _enabled = false;
        private int _threshold = 127;
        //>>

        //<<ctor
        public BinarizationFilter()
        {
            AddBoolParam( "enable", new SetBoolParam( setEnable ), new GetBoolParam( getEnable ) );
            AddIntParam( "threshold", new SetIntParam( setThreshold ), new GetIntParam( getThreshold ) );
        }
        //>>

        /*
         *	Enables or disables the binarization.
         *
         *	Only call this method in a beginParamTransfer/endParamTransfer block.
         */
        //<<setenable
        void setEnable( bool enable )
        {
            _enabled = enable;
        }
        //>>

        /*
         *	Get the current enabled state of the binarization filter.
         *
         *	Only call this method in a beginParamTransfer/endParamTransfer block.
         */
        //<<getenable
        bool getEnable()
        {
            return _enabled;
        }
        //>>

        /*
         *	Sets the threshold value for the binarization.
         *
         *	Only call this method in a beginParamTransfer/endParamTransfer block.
         */
        //<<setthreshold
        void setThreshold( int threshold )
        {
            _threshold = threshold;
        }
        //>>

        /*
         *	Get the current threshold value of the binarization filter.
         *
         *	Only call this method in a beginParamTransfer/endParamTransfer block.
         */
        //<<getthreshold
        int getThreshold()
        {
            return _threshold;
        }
        //>>

        /*
         * This method fills the ArrayList arr with the frame types this filter
         * accepts as input.
         * 
         * For the binarization filter, only the gray color formats eY800 and eRGB8 are accepted.
         */
        //<<getSupportedInputTypes
        public override void GetSupportedInputTypes( System.Collections.ArrayList frameTypes )
        {
            // This filter works for 8-bit-gray images only
            frameTypes.Add( new FrameType(MediaSubtypes.Y800 ) );
        }
        //>>

        /*
         *	This method returns the output frame type for a given input frame type.
         *
         *	The binarization filter does not change size or color format,
         *	so the only output frame type is the input frame type.
         */
        //<<getTransformOutputTypes
        public override bool GetTransformOutputTypes( FrameType inType, System.Collections.ArrayList outTypes )
        {
            // We don't change the image type, output = input
            outTypes.Add( inType );

            return true;
        }
        //>>

        /*
         *	This method is called to copy image data from the src frame to the dest frame.
         *
         *	Depending on the value of m_bEnabled, this implementation applies a binarization or
         *	copies the image data without modifying it.
         */
        //<<transform
        public override bool Transform( IFrame src, IFrame dest )
        {
            unsafe
            {
                // Check whether the destination frame is available
                if( dest.Ptr == null ) return false;

                // Copy the member variables to the function's stack, to protect them from being
                // overwritten by parallel calls to setThreshold() etc.
                //
                // beginParamTransfer/endParamTransfer makes sure that the values from various
                // member variables are consistent, because the user of this filter must enclose
                // writing parameter access into beginParamTransfer/endParamTransfer, too.

                BeginParameterTransfer();
                int threshold = _threshold;
                bool enabled = _enabled;
                EndParameterTransfer();

                byte* pIn = src.Ptr;
                byte* pOut = dest.Ptr;

                // Check whether binarization is enabled
                if( enabled )
                {
                    // For each byte in the input buffer, check whether it is greater or
                    // equal to the threshold.
                    int bufferSize = src.FrameType.BufferSize;
                    while( bufferSize-- > 0 )
                    {
                        if( *pIn++ >= threshold )
                        {
                            *pOut++ = 255;
                        }
                        else
                        {
                            *pOut++ = 0;
                        }
                    }
                }
                else
                {
                    // Binarization is disabled: Copy the image data without modifying it.
                    dest.CopyFrom( src );
                }
            }

            return true;
        }
        //>>
    }

}
