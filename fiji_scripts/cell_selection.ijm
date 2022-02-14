// ImageJ 1 Macro to select rectangles in batch
// Inspired by http://imagej.1557.x6.nabble.com/Create-pre-defined-ROI-on-mouse-click-and-add-to-ROI-manager-tp5008292p5008310.html
// Modified by MW, Nov 2019, GPLv3+

macro selectRegion {
    setOption("DisablePopupMenu", true); // No idea what this does
    getPixelSize(unit, pixelWidth, pixelHeight);
    setTool("rectangle");
    leftButton=16;
    rightButton=4;

    // Important variables
    winCrop = 300;
    sd1=0.8;
    sd2=10.0;

    // Dialog to select parameters
    Dialog.create("Settings");
    Dialog.addNumber("Set crop size", winCrop);
    Dialog.addNumber("Blur radius 1", sd1);
    Dialog.addNumber("Blur radius 2", sd2);
    Dialog.addMessage("After clicking OK,\n click on the center of the cell.\n Right click when done.");
    Dialog.show();

    // Retrieve dialog variables
    winCrop = Dialog.getNumber();
    sd1 = Dialog.getNumber();
    sd2 = Dialog.getNumber();

    // Logic to acquire the coordinates of the rectangle
    x2=-1; y2=-1; z2=-1; flags2=-1;
    getCursorLoc(x, y, z, flags);
    wasLeftPressed = false;
    while (flags&rightButton==0){
            getCursorLoc(x, y, z, flags);
            if (flags&leftButton!=0) {
            // Wait for it to be released
            wasLeftPressed = true;
            } else if (wasLeftPressed) {
            wasLeftPressed = false;
            if (x!=x2 || y!=y2 || z!=z2 || flags!=flags2) {
                            xc = x - winCrop/2;
                            yc = y - winCrop/2;
                            xx = x;
                            yy = y;
                            makeRectangle(xc, yc, winCrop, winCrop);
                            //roiManager("Add");
                             
                    }
            }
    }

    // Logic to acquire the coordinates of the pillar
    //Dialog.create("Settings");
    //Dialog.addMessage("Now position the pillar with the rotated rectangle\nAfter clicking OK\n Right click when done.");
    //Dialog.show();

    setOption("DisablePopupMenu", true); // No idea what this does
    setTool("rotrect");
    makeRotatedRectangle(214, 268, 686, 722, 647);    
    waitForUser("Now position the pillar with the rotated rectangle\nClick OK when done.");
    //getCursorLoc(xX, yY, zZ, flagsS);
    //IJ.log(flagsS&rightButton);
    //IJ.log(flagsS);
    //IJ.log(rightButton);
    //while (flagsS&rightButton==0||flagsS==36){
    //    getCursorLoc(xX, yY, zZ, flagsS);
    //}

    // Extract rotated rectangle coordinates
    getSelectionCoordinates(xpoints, ypoints);
    xr="rect = ";
    for (i=0;i<xpoints.length;i++) {
	xr=xr+"("+xpoints[i]+","+ypoints[i]+"),";
    }

    // Logic to acquire the coordinates of the locus
    setTool("point");
    waitForUser("Now click on the locus at the first point after the magnet was added.\n Click okay when done.");
    getSelectionCoordinates(x0,y0);
    IJ.log("x0="+x0[0]);
    IJ.log("y0="+y0[0]);
        
    // Display results
    setOption("DisablePopupMenu", false);
    dir = getDirectory("image"); 
    name = getTitle; 
    Stack.getPosition(channel, slice, frame);
    getPixelSize(unit, pixelWidth, pixelHeight); 
    getVoxelSize(width, height, depth, unit);

    Dialog.create("Results");
    Dialog.addMessage("The following will be saved");
    Dialog.addString("Path", dir+name+".txt");
    Dialog.addNumber("x", xx);
    Dialog.addNumber("y", yy);
    Dialog.addNumber("z", slice);
    Dialog.addNumber("DNA channel (1-based)", channel);
    Dialog.addNumber("XY pixel size", width);
    Dialog.addNumber("Z spacing", depth);
    Dialog.addNumber("Locus x0", x0[0]);
    Dialog.addNumber("Locus y0", y0[0]);

    Dialog.addCheckbox("Open next file in line...", false);
    Dialog.show();

    // Save
    of = Dialog.getString();
    xx = Dialog.getNumber();
    yy = Dialog.getNumber();
    z = Dialog.getNumber();
    channel = Dialog.getNumber();
    width = Dialog.getNumber();
    depth = Dialog.getNumber();
    xx0 = Dialog.getNumber();
    yy0 = Dialog.getNumber();
    
    xr="rect = ";
    for (i=0;i<xpoints.length;i++) {
	xr=xr+"("+xpoints[i]+","+ypoints[i]+"),";
    }
    
    IJ.log("Saving to: "+of);
    IJ.log(xr);    

    File.append("["+name+"]", of);
    File.append("winCrop = "+winCrop, of);
    File.append("sd1 = "+ sd1, of);
    File.append("sd2 = "+ sd2, of);
    File.append("fn = "+ dir+name, of);
    File.append("xLoc = "+ xx, of);
    File.append("yLoc = "+ yy, of);
    File.append("zLoc = "+ z, of);
    File.append("chDNA = "+ channel, of);
    File.append("pixXY = "+ width, of);
    File.append("stepZ = "+ depth, of);
    File.append(xr, of);
    File.append("x0 = " + xx0, of);
    File.append("y0 = " + yy0, of);
    File.append("", of);

    if (Dialog.getCheckbox()) {
  	run("Open Next");
    }  
} 
