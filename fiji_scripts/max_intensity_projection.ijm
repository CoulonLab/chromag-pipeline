// Script to extract a max intensity projection from a 5D movie
// By MW, GPLv3+, Aug 2020

write("Found "+inF.length+" files");

for (i=0; i<inF.length ; i++) {
    write((i+1)+"/"+inF.length+": Performing max-intensity projection from: "+ inF[i]+ " to: "+ ouF[i]);
    open(inF[i]);
    run("Z Project...", "projection=[Max Intensity] all");
    saveAs("Tiff", ouF[i]);
    close();
    close(); // we progressively close the images, to avoid memory problems
}
write("-- All done!");