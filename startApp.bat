@echo off
echo Cleaning old builds...
rmdir /s /q dist build

echo Building with PyInstaller...
pyinstaller --onedir --console --clean --collect-all scipy --collect-all numpy --collect-all pandas --collect-all torch --collect-all torchvision --collect-all cv2 --collect-all sklearn --collect-all skimage --collect-all sam2 --collect-all PIL --collect-all matplotlib --collect-all seaborn --add-data "sam2;sam2" --add-data "ImageAnalysis;ImageAnalysis" --add-data "ImagePreprocessing;ImagePreprocessing" --hidden-import "Config_Manager" --hidden-import "ProcessStartModel" --hidden-import "ImageAnalysisModel" --hidden-import "ImageProcessingModel" --hidden-import "ParticleSegmentationModel" --hidden-import "CalibrationModel" --hidden-import "ContainerScalerModel" --hidden-import "pyDeepP2SA" --hidden-import "sizeAnalysisModel" --hidden-import "getCheckpoints" --hidden-import "ROISelector" startModel.py

echo Copying required folders...
xcopy "ImageAnalysis" "dist\startModel\ImageAnalysis" /E /I
xcopy "ImagePreprocessing" "dist\startModel\ImagePreprocessing" /E /I

echo Build completed!