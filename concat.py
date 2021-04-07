#-*- coding:utf-8 -*-
## Fiji Macro to concatenate stacks
## Takes the paths to concatenate as input arguments
import sys, os
from ij import IJ
import ij.gui
from java.awt import Font
from ij.plugin import Duplicator, Concatenator, SubHyperstackMaker
from loci.plugins import BF
from loci.plugins.in import ImporterOptions, ImportProcess
opt = ImporterOptions()
opt.setVirtual(False)

## Constants
useBF=False ## Set to True to open files using Bio-Formats

types = {ij.ImagePlus.COLOR_RGB : "RGB",  
         ij.ImagePlus.GRAY8 : "8-bit",  
         ij.ImagePlus.GRAY16 : "16-bit",  
         ij.ImagePlus.GRAY32 : "32-bit",  
         ij.ImagePlus.COLOR_256 : "8-bit color"}

## === File checks
lf = sys.argv[1:-2]
of = sys.argv[-2] # output file
tf = sys.argv[-1] # Address of the file of timestamps

## DBG
#lf = ['/data/CoulonLab/CoulonLab Dropbox/data/Laura/20190415/20190415_u2os_smallarray_tetRmCherry_sirDNA_Zstack_Injections_trials_5/20190415_u2os_smallarray_tetRmCherry_sirDNA_Zstack_Injections_trials_5_MMStack_Pos0.ome.tif','/data/CoulonLab/CoulonLab Dropbox/data/Laura/20190415/20190415_u2os_smallarray_tetRmCherry_sirDNA_Zstack_Injections_trials_6/20190415_u2os_smallarray_tetRmCherry_sirDNA_Zstack_Injections_trials_6_MMStack_Pos0.ome.tif']
print "Concatenating %i files" % len(lf)
print "Files to concatenate:"
for i in lf:
    print "-- %s" % i
    if not os.path.isfile(i):
        print "File NOT FOUND, aborting!"
        sys.exit(1)

## === Open files
imL = []
if not useBF: ## DBG
    for i in lf:
        imp = IJ.openImage(i)
        imL.append(imp)
else:
    for i in lf:
        opt.setId(i)
        process = ImportProcess(opt)
        process.execute()
        ns = -1
        for j in range(process.getSeriesCount()):
            if process.getSeriesLabel(j).split(": ")[1]==os.path.basename(i).split(".")[0]:
                ns=j
        if ns==-1 and process.getSeriesCount()==1:
            print "Weird..."
            ns=0
        else:
            pn = i.split("_Pos")[-1].split(".")[0]
            print "Inferred position number:", pn
            for j in range(process.getSeriesCount()):
                if "_Pos"+str(int(pn)) in process.getSeriesLabel(j):
                    ns=j
                    break
            if ns==-1:
                print process.getSeriesCount()
                print [process.getSeriesLabel(i) for i in range(process.getSeriesCount())]
                print "GRAVE ERROR!"
                sys.exit(1)
        opt.setSeriesOn(0,False)
        opt.setSeriesOn(ns,True)
        print "-- Opening "+process.getSeriesLabel(ns)
        im = BF.openImagePlus(opt)
        opt.setSeriesOn(ns,False)
        imp = im[-1]
        imL.append(imp)

for i in imL:
    print i.getTitle(), i.getNSlices(), i.getNChannels(), i.getNFrames()

## === Add z planes
maxZ = max([i.getNSlices() for i in imL])
maxC = max([i.getNChannels() for i in imL])
ssL = [] # Same number of slices
nff=True
for im in imL:
    nz = maxZ-im.getNSlices() # Number of z planes to add
    if nz==0 and im.getNChannels()==maxC:
        ssL.append(im)
        #pix = im.getProcessor().convertToFloat().getPixels()
        # find out the minimal pixel value
        #MI = reduce(max, pix)
        #print "Case1", MI
        continue
    if im.getNFrames()==1:
        tmp = Duplicator().run(im, 1, im.getNChannels(), 1, 1, 1, 1) # create a duplicated slice
        tmp.getProcessor().multiply(0) # make it a dark frame    
        ssL.append(Concatenator().concatenate([im]+[tmp]*nz, False)) # This deletes references/images in imL
        #print "Case 2"
    else: # if im.getNFrames()>1 (we need to concatenate time-point-per timepoint :s )
        nff=False        
        allFrames = []
        stack = im.getImageStack()
        nZ = im.getNSlices()
        nC = im.getNChannels()
        dark = Duplicator().run(im, 1, 1, 1, 1, 1, 1)
        dark.getProcessor().multiply(0)
        res = ij.ImageStack(im.width, im.height)
        icnt = 0
        for ia in range(1, im.getNFrames()+1):
            for ib in range(1, maxZ+1):
                for ic in range(1,maxC+1):
                    icnt +=1
                    if (ib>im.getNSlices()) or (ic>im.getNChannels()):
                        res.addSlice(str(icnt), dark.getProcessor())
                        print ic, "/", im.getNFrames(), index, (ia-1)*nZ*nC+index, "/", stack.getSize(), "*"
                    else:
                        index = im.getStackIndex(ic, ib, ia)
                        print index
                        print ic, "/", im.getNFrames(), index, (ia-1)*nZ*nC+index, "/", stack.getSize(), stack.getProcessor(index).getMax()
                        #res.addSlice(str(icnt), stack.getProcessor((ia-1)*nZ*nC+index))
                        res.addSlice(str(icnt), stack.getProcessor(index))
        im2 = ij.ImagePlus('title', res)
        im2 = ij.plugin.HyperStackConverter().toHyperStack(im2,nC, maxZ, im.getNFrames())
        #ij.plugin.HyperStackConverter().shuffle(im2, 5)
        ssL.append(im2)

print "New dimensions:"
nf = 0
for (j,i) in enumerate(ssL):
    nf+=i.getNFrames()
    print "%i/%i" % (j+1, len(ssL)), i.getNSlices(), i.getNChannels(), i.getNFrames(), i.getTitle()
        
print "Concatenating everything"
out = Concatenator().concatenate(ssL, False)
if nff:
    out = ij.plugin.HyperStackConverter.toHyperStack(out,out.getNChannels(), maxZ, nf)
if out is None:
    print "Concatenation failed"
    sys.exit(1)
print out.getNChannels(), out.getNFrames(), out.getNSlices(), types[out.type]
print out.getNChannels(), nf, maxZ
print "Stacks concatenated, dimensions: (S=%i, C=%i, F=%i), adding overlay." % (out.getNSlices(), out.getNChannels(), out.getNFrames())
#IJ.saveAs(out, "Tiff", of+'.DEUX.ome.tif') ## DBG

# === Identify & remove dark frames
def extractFrame(imp, nFrame):
 """ Extract a stack for a specific color channel and time frame """
 stack = imp.getImageStack()
 ch = ij.ImageStack(imp.width, imp.height)
 for i in range(1, imp.getNSlices() + 1):
     for j in range(1, imp.getNChannels()+1):
         index = imp.getStackIndex(j, i, nFrame)
         ch.addSlice(str(i)+"-"+str(j), stack.getProcessor(index))
 return ij.ImagePlus("Frame " + str(nFrame), ch)

toDel = []
for i in range(out.getNFrames()):
    f=extractFrame(out, i+1)
    if f.getStatistics().max==0:
        print "Frame {} is dark, skipping".format(i)
        toDel.append(i+1)
toKeep = [i for i in range(1, out.getNFrames()+1) if i not in toDel]
out = SubHyperstackMaker().makeSubhyperstack(out,
                                             range(1, out.getNChannels()+1),
                                             range(1, out.getNSlices()+1),
                                             toKeep)

# === Apply overlay
# Inspired from https://forum.image.sc/t/text-overlay-in-python-for-imagej/21989/4
# And: https://forum.image.sc/t/displaying-slices-overlays-in-a-stacked-image/3700/4
f=open(tf, 'r')
ts=[i.replace('\n', '') for i in f.readlines()]
f.close()

## Check
if out.getNFrames() != len(ts):
    print "Either we are missing some timestamps (there are {}) or there are extra frames (there are {}) in the data, ABORTING".format(len(ts), out.getNFrames())
    sys.exit(1)

## Add overlay
font = Font("SanSerif", Font.PLAIN, 12)
overlay = ij.gui.Overlay()

for (i,t) in enumerate(ts):
    roi = ij.gui.TextRoi(10, 5, "%s" % t, font)
    roi.setPosition(0,0,i+1) # order: (channel, slice, frame), 0 means all.
    overlay.add(roi)
out.setOverlay(overlay)

## Save pixel sizes
c=imL[0].getCalibration()
try:
    un = c.getUnit().decode('ascii')
except UnicodeDecodeError: # unit is micron
    un = "micron"

IJ.run(out, "Properties...", "unit={} pixel_width={} pixel_height={} voxel_depth={}".format(un, c.pixelWidth, c.pixelHeight, c.pixelDepth))

print "Overlay & pixel sizes added. Saving."

# === Save
if os.path.isfile(of):
    print "Output file already exists, ABORTING: %s" % of
else:
     IJ.saveAs(out, "Tiff", of)
print "Done!"
