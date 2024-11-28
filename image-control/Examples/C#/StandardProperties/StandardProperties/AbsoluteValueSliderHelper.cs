
using System;
using TIS.Imaging;

namespace TIS.Imaging.Helper
{
    static class AbsoluteValueSliderHelper
    {
        private static int AbsValToSliderPosMapped( VCDAbsoluteValueProperty itf, double val, int sliderSteps, System.Func<double, double> map )
        {
            double range_min = itf.RangeMin;
            double range_max = itf.RangeMax;

            double mapped_min = map(range_min);
            double mapped_max = map(range_max);
            double mapped_val = map(val);

            double mapped_range = mapped_max - mapped_min;
            double mapped_offset = mapped_val - mapped_min;

            return (int)(sliderSteps * mapped_offset / mapped_range + 0.5);
        }

        private static double SliderPosToAbsValMapped( VCDAbsoluteValueProperty itf, int pos, int sliderSteps, System.Func<double, double> map, System.Func<double, double> unmap )
        {
            double range_min = itf.RangeMin;
            double range_max = itf.RangeMax;

            double mapped_min = map(range_min);
            double mapped_max = map(range_max);

            double mapped_range = mapped_max - mapped_min;

            double mapped_val = mapped_min + mapped_range * pos / sliderSteps;

            return unmap(mapped_val);
        }

        public static int AbsValToSliderPosLogarithmic( VCDAbsoluteValueProperty itf, double val, int sliderSteps )
        {
            return AbsValToSliderPosMapped(itf, val, sliderSteps, ( x ) => Math.Log(x));
        }

        public static double SliderPosToAbsValLogarithmic( VCDAbsoluteValueProperty itf, int pos, int sliderSteps )
        {
            return SliderPosToAbsValMapped(itf, pos, sliderSteps, ( x ) => Math.Log(x), ( x ) => Math.Exp(x));
        }

        public static int AbsValToSliderPosLinear( VCDAbsoluteValueProperty itf, double val, int sliderSteps )
        {
            return AbsValToSliderPosMapped(itf, val, sliderSteps, ( x ) => x);
        }

        public static double SliderPosToAbsValLinear( VCDAbsoluteValueProperty itf, int pos, int sliderSteps )
        {
            return SliderPosToAbsValMapped(itf, pos, sliderSteps, ( x ) => x, ( x ) => x);
        }

    }
}
