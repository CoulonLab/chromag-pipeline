## Pre-processing for drift correction in Jython (Fiji+Python)
## DO NOT EDIT THIS CELL!!!
## By MW, GPLv3+, Nov 2019
## Be careful, Jython is Python 2.5...

"""
The input parameters are defined as:
- `fName`  -> raw data file name
- `chDNA`  -> channel number (1-based) of DNA dye
- `pilTop` -> put 1 for pillar top or 0 for pillar bottom
- `suff`   -> suffix for file naming,
- `xLoc, yLoc, zLoc` -> x, y z position of the locus on 1st frame (in pixel/slice),
- `xShift, yShift, zShift` -> max shifts in x, y and z (from output of drift correction. Put 0 if unknown)
"""

import math, sys, ConfigParser, ast, os

#os.environ['DISPLAY']=':50.0' # We need an active -X display somewhere.
print "Using display on {}".format(os.environ['DISPLAY'])

from collections import OrderedDict
import ij, ij.process, ij.gui
from ij import IJ
from java.awt import Font
from ij.plugin import Duplicator, ImageCalculator
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
sys.path.append("/data/CoulonLab/Maxime/chromag-pipeline/")
import Correct_3D_Drift as correctDrift



IJ.log("Pre-processing for drift correction")

## ==== Variables
fn = sys.argv[-4]  # Input file should be read from command-line
of = sys.argv[-3]
cellId = int(sys.argv[-2]) # 0-based
correction_type = sys.argv[-1] # Should be a string
winDrift=150
crop_image = True
followLocus = False
if correction_type == 'none':
    run_DC = False
elif correction_type == 'add_bp':
    run_DC = False
    crop_image = False
else:
    run_DC  = True
    if correction_type == 'followLocus':
        followLocus = True

## ==== Calculated variables
if fn == of:
    print "Output file is the same as the input file, aborting"
    sys.exit(1)

fn_cfg = fn+'.txt'  # Location of the config file

# xShift, yShift ## initially to crop the data

## ==== Functions
class multidict(OrderedDict):
    """Rename duplicated sections so that they can be imported separately
    From: https://stackoverflow.com/a/9888814"""
    _unique = 0   # class variable

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += 1
            key += str(self._unique)
        OrderedDict.__setitem__(self, key, val)

## === Load ConfigFile
IJ.log("Opening config file at %s" % sys.argv[1])
parser = ConfigParser.ConfigParser(defaults=None, dict_type=multidict)#, strict=False)
parser.read(fn_cfg)
section = parser.sections()[cellId]  # Select the section corresponding to the cell

winCrop = parser.getint(section, 'winCrop')
sd1 = parser.getfloat(section, 'sd1')
sd2 = parser.getfloat(section, 'sd2')
#fn = parser.get(section, 'fn')
xLoc = parser.getint(section, 'xLoc')
yLoc = parser.getint(section, 'yLoc')
zLoc = parser.getint(section, 'zLoc')
chDNA = parser.getint(section, 'chDNA')
pixXY = parser.getfloat(section, 'pixXY')
stepZ = parser.getfloat(section, 'stepZ')
rect = ast.literal_eval(parser.get(section, 'rect'))

if followLocus:
    if chDNA>1:
        chDNA-=1
    else:
        chDNA=2

## === Open as a BioFormats image stack
## Inspired from: https://forum.image.sc/t/virtual-stack-bioformats-macro-command/23134
#of = "/data/CoulonLab/CoulonLab Dropbox/data/Maxime/Laura/20190415/20191107/concatenated_Pos27DC.ome.tif"
#fn = "/data/CoulonLab/CoulonLab Dropbox/data/Maxime/Laura/20190415/20191107/concatenated_Pos27.ome.tif"
opt = ImporterOptions()
opt.setVirtual(True)
opt.setId(fn)
im = BF.openImagePlus(opt)
imp = im[0]
if imp is None:  
      print "Could not open image from file:"
IJ.log("title: %s" % imp.title)  
IJ.log("width: %i" % imp.width)
IJ.log("height: %i" % imp.height)
IJ.log("number of slices: %i" % imp.getNSlices())
IJ.log("number of channels: %i" % imp.getNChannels())
IJ.log("number of time frames: %i" % imp.getNFrames())
IJ.log("the channel to track is channel %i" % chDNA)
  
types = {ij.ImagePlus.COLOR_RGB : "RGB",  
         ij.ImagePlus.GRAY8 : "8-bit",  
         ij.ImagePlus.GRAY16 : "16-bit",  
         ij.ImagePlus.GRAY32 : "32-bit",  
         ij.ImagePlus.COLOR_256 : "8-bit color"}  
  
IJ.log("image type: %s" % types[imp.type])

## ==== Preprocess the image
imp.getCalibration().pixelWidth = pixXY
imp.getCalibration().pixelHeight = pixXY
imp.getCalibration().pixelDepth = stepZ
#imp.getCalibration().getUnit() # should return 'micron'
#imp.getCalibration() # DBG
if crop_image:
    imp.setRoi(int(xLoc-math.floor(winCrop/2)),int(yLoc-math.floor(winCrop/2)),winCrop,winCrop) # Create a ROI and crop it.
else:
    imp.setRoi(0,0,imp.width,imp.height) # Create a ROI and crop it.

crop = imp.crop("stack")
ij.process.ImageConverter(crop).convertToGray32()
tmp1 = Duplicator().run(crop, chDNA, chDNA, 1, crop.getNSlices(), 1, crop.getNFrames()) #Stack.setChannel(chDNA); run("Duplicate...", "channels=2 title=cropDNA duplicate");
tmp2 = Duplicator().run(crop, chDNA, chDNA, 1, crop.getNSlices(), 1, crop.getNFrames()) #run("Duplicate...", "title=tmp2 duplicate");

# === Band pass filtering
IJ.run(tmp1, "Gaussian Blur...", "sigma=%f stack" % sd1) #selectWindow("tmp1"); run("Gaussian Blur...", "sigma="+sd1+" stack");
IJ.run(tmp2, "Gaussian Blur...", "sigma=%f stack" % sd2) #selectWindow("tmp2"); run("Gaussian Blur...", "sigma="+sd2+" stack");
cropDNA_bPass = ImageCalculator().run("Subtract create 32-bit stack", tmp1, tmp2) #imageCalculator("Subtract create 32-bit stack", "tmp1","tmp2"); rename("cropDNA_bPass");
tmp1.close()
tmp2.close()

# === Create stack to be processed
# From https://stackoverflow.com/questions/48213759/combine-channels-in-imagej-jython
crop_channels = ij.plugin.ChannelSplitter.split(crop) # selectWindow("crop"); run("Split Channels");
if len(crop_channels)>=2:
    res_st = ij.plugin.RGBStackMerge().mergeHyperstacks([cropDNA_bPass, ]+[c for c in crop_channels], True) #run("Merge Channels...", "c1=cropDNA_bPass c2=C1-crop c3=C2-crop create")
else :
    IJ.log("NOT IMPLEMENTED ERROR")
    raise NotImplementedError

# === 3D drift correction (we are working with res_st as input image)
res_st.setRoi(int(winCrop/2)-int(winDrift/2), int(winCrop/2)-int(winDrift/2), winDrift, winDrift)
if run_DC:
    dc_st = correctDrift.run_cli(res_st, only_compute=False, multi_time_scale=False, verbose=False)
    out = dc_st[1]
else:
    out = res_st

# === Apply overlay
# Inspired from https://forum.image.sc/t/text-overlay-in-python-for-imagej/21989/4
# And: https://forum.image.sc/t/displaying-slices-overlays-in-a-stacked-image/3700/4
if not os.path.isfile(fn+'.time'):
    print "TIMESTAMP file not found"
else:
    f=open(fn+'.time', 'r')
    ts=[i.replace('\n', '') for i in f.readlines()]
    f.close()

    font = Font("SanSerif", Font.PLAIN, 12)
    overlay = ij.gui.Overlay()

    for (i,t) in enumerate(ts):
        roi = ij.gui.TextRoi(10, 5, "%s" % t, font)
        roi.setPosition(0,0,i+1) # order: (channel, slice, frame), 0 means all.
        overlay.add(roi)
        out.setOverlay(overlay)

# === Saving
IJ.log("Saving.")
#print type(dc_st), dc_st
if not os.path.isfile(of):
    IJ.save(out, of)
    #if dc_st is None:
    #    f=open(of, 'w')
    #    f.write("Only one frame\n")
    #    f.close()
    #else:
    #    IJ.save(out, of)
# === Save shifts (if needed)
if run_DC:
    f=open(of+'.shifts', "w")
    f.write("frame,x,y,z\n")
    for (i,s) in enumerate(dc_st[0]):
        f.write("%i, %f, %f, %f\n" % (i,s.x,s.y, s.z))
    f.close()
    
IJ.run("Quit")
