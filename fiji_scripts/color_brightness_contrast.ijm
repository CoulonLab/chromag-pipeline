Stack.setChannel(1);
resetMinAndMax();
run("Green");
run("Next Slice [>]");
run("Grays");
resetMinAndMax();
Stack.setActiveChannels("01100");
//run("Z Project...", "projection=[Max Intensity] all");
//run("RGB Color", "frames keep");