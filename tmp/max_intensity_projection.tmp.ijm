inF=newArray("/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos3DC0.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos3DC1.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos3DC2.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos2DC0.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos2DC1.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos1DC0.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos1DC1.ome.tif");
ouF=newArray("/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos3DC0_MAX.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos3DC1_MAX.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos3DC2_MAX.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos2DC0_MAX.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos2DC1_MAX.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos1DC0_MAX.ome.tif", "/data2/Dropbox/CoulonLab/data/Maxime/drift-correction/Antoine/20200221/20220214/concatenated_Pos1DC1_MAX.ome.tif");

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