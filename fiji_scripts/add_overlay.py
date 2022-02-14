## A simple script to add an overlay to a movie, in Python 3
## By MW, GPLv3+, May 2020

import os, sys
import numpy as np
import pandas as pd
from skimage.external.tifffile import TiffFile

## Fiji/Java imports
## == Environment, to get stuff working. Be careful, this easily breaks.
os.environ['PYJNIUS_JAR'] = "/data/CoulonLab/Maxime/jupyter/jupyter/share/pyjnius/pyjnius2.jar"

import imagej
fiji_path = '/home/umr3664/.Fiji.app'
assert os.path.isdir(fiji_path)
ij = imagej.init(fiji_path, headless=False)
#ij = imagej.init('sc.fiji:fiji') # We import the latest Fiji version, to be reproducible.
#ij = imagej.init('net.imagej:imagej+net.imagej:imagej-legacy', headless=False)

import jnius
from jnius import autoclass
print("jnius version: {}".format(jnius.__version__)) # Version 1.2.1 breaks everything so far, stick to 1.2.0
class IjGui: # Dummy object for storage
    def __init__(self):
        pass

Font = autoclass("java.awt.Font")
ij_gui = IjGui()
ij_gui.Overlay = autoclass("ij.gui.Overlay")
ij_gui.TextRoi = autoclass("ij.gui.TextRoi")
ij_gui.Line = autoclass("ij.gui.Line")
ij_gui.Arrow = autoclass("ij.gui.Arrow")
ij_gui.Color = autoclass("java.awt.Color")

ImagePlus = autoclass("ij.ImagePlus")
IJ = autoclass("ij.IJ")
ImageJFunctions = autoclass("net.imglib2.img.display.imagej.ImageJFunctions")   

def apply_overlay(in_i, tb, norm, verbose=False):
    """Export regions with arrows corresponding to the direction of the force and the motion
    in_i: input image on which we need to add an overlay
    tb (dataframe): a dataframe with timestamp information
    norm (bool): whether the intensity of the siRDNA channel should be normalized
    based on the ALLCELLS_mean... column"""
    
    ## Sanity checks
    hasZ = True
    if len(in_i.shape)<5:
        hasZ=False

    ## Normalize if needed
    if norm:
        #no_f = {int(i.frame):i.ALLCELLS_mean_avg_norm for (ii,i) in tb.iterrows()}
        no_f = {int(i.frame):i.ALLCELLS_std_avg_norm for (ii,i) in tb.iterrows()}
        for i in range(in_i.shape[0]):
            if hasZ and i in no_f:
                in_i[i, :, 2]=(in_i[i, :, 2]*no_f[i]).astype(np.int16)
            elif not hasZ and i in no_f:
                in_i[i, 2]=(in_i[i, 2]*no_f[i]).astype(np.int16)
        
    ## Transfer data
    rai = ij.py.to_java(in_i)
    out = ImageJFunctions.wrap(rai, 'image')

    font = Font("SanSerif", Font.PLAIN, 16)
    overlay = ij_gui.Overlay()
    for (i,t) in tb.iterrows():
        
        roi = ij_gui.TextRoi(10, 5, "{:.1f}s ({})".format(
            t.seconds_since_first_magnet_ON,
            ['OFF', 'ON'][t.forceActivated]), font)
        if hasZ:
            roi.setPosition(0,0,int(t.t)+1) # order: (channel, slice, frame), 0 means all.
        else:
            roi.setPosition(0,int(t.t)+1,0)
        overlay.add(roi)
    out.setOverlay(overlay)
    return out

def apply_display(imp):
    imp.setDisplayMode(IJ.COMPOSITE);
    imp.setC(2);
    IJ.setMinAndMax(imp, 0, 30000)
    #IJ.run("Green")

    imp.setC(3);
    IJ.resetMinAndMax(imp)
    IJ.run("Grays")

    imp.setZ(10)

    imp.setActiveChannels("01100");
    return imp

def write_file(out, out_n):
    assert not os.path.isfile(out_n), "The output file already exists {}. Aborting.".format(out_n)
    IJ.saveAs(out, 'Tiff', out_n)
    
if __name__ == "__main__":
    ## Parse arguments
    import sys
    in_p = sys.argv[1]
    out_p = sys.argv[2]
    tb_p = sys.argv[3]
    if len(sys.argv)>4 and sys.argv[4]=="norm":
        norm=True
    else:
        norm=False
    
    assert os.path.isfile(tb_p)
    tb = pd.read_csv(tb_p)
    in_i = TiffFile(in_p).asarray() ## Load the movie

    if norm:
        out_p = out_p.replace('.tif', '_norm.tif')

    assert not os.path.isfile(out_p), "The output file {} already exists. Aborting.".format(out_n)
    im = apply_overlay(in_i, tb, norm)
    out = apply_display(im)
    write_file(out, out_p)
    print("--Done!")
