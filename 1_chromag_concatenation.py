#!/usr/bin/env python
# coding: utf-8

# ## Step 1 - concatenation
# 
# *The code in this repository is shared under the GPLv3+ license, by Maxime Woringer, Mar. 2021*
# 
# ### input data
# A Micromanager folder, in which several positions were imaged multiple times, leading to a folder structure as follows:
# 
# ```
# ├── 20200221_release_1
# │   ├── 20200221_release_1_MMStack_Pos1.ome.tif
# │   ├── 20200221_release_1_MMStack_Pos2.ome.tif
# │   ├── 20200221_release_1_MMStack_Pos3.ome.tif
# │   ├── 20200221_release_1_MMStack_Pos4.ome.tif
# │   ├── 20200221_release_1_MMStack_Pos5.ome.tif
# │   └── displaySettings.txt
# ├── 20200221_beforeattr_2
# │   ├── 20200221_beforeattr_2_MMStack_Pos1.ome.tif
# │   ├── 20200221_beforeattr_2_MMStack_Pos2.ome.tif
# │   ├── 20200221_beforeattr_2_MMStack_Pos3.ome.tif
# │   ├── 20200221_beforeattr_2_MMStack_Pos4.ome.tif
# │   ├── 20200221_beforeattr_2_MMStack_Pos5.ome.tif
# │   └── displaySettings.txt
# ├── 20200221_beforeattr_3
# │   ├── 20200221_beforeattr_3_MMStack_Pos1.ome.tif
# │   ├── 20200221_beforeattr_3_MMStack_Pos2.ome.tif
# │   ├── 20200221_beforeattr_3_MMStack_Pos3.ome.tif
# │   ├── 20200221_beforeattr_3_MMStack_Pos4.ome.tif
# │   ├── 20200221_beforeattr_3_MMStack_Pos5.ome.tif
# │   └── displaySettings.txt
# ├── 20200221_beforeattr_4
# │   ├── 20200221_beforeattr_4_MMStack_Pos1.ome.tif
# │   └── displaySettings.txt
# ```
# 
# ### output data
# This scripts creates one movie per position (named with the position number), and recovers the timestamps of the individual frames. The timestamps are exported as a `.xls` file, and overlaid in the concatenated file. 
# Finally, if the timestamp when the magnet was added/removed is present in the configuration (`.cfg`) file, then *(ON|OFF)* flag is present both in the `.xls` file and in the movie overlay.
# 
# ```
# ├── concatenated_Pos1.ome.tif
# ├── concatenated_Pos2.ome.tif
# ├── concatenated_Pos3.ome.tif
# ├── concatenated_Pos4.ome.tif
# ├── concatenated_Pos5.ome.tif
# ├── files_concatenated.txt
# ├── timestamps_Pos1.xls
# ├── timestamps_Pos2.xls
# ├── timestamps_Pos3.xls
# ├── timestamps_Pos4.xls
# └── timestamps_Pos5.xls
# ```
# 
# ### parameters of this script
# This script takes very little parameters, they are located in two `.cfg` files in the `config` folder, and in the first cell below.
# 
# #### `datasets.cfg`
# 
# Datasets are represented as sections ; sections are delimited by headers [section_name]. Each section should contain:
# - A `lfn` variable, should be a Python list that contains the list of folders to include for the concatenation. This allows including/excluding folders that should/should not be concatenated
# - A `forceOn` variable, a list of list, each inner list should contain the timestamp when the magnet was added, the removed. The list can contain multiple lists if the magnet was added/removed several time.
#   - Example 1: magnet added at 16:00:00, removed at 16:30:00: `forceOn = [['2020-02-21 16:00:00', '2020-02-21 16:30:00']]`
#   -  Example 2: in addition the magnet was re-added at 17:00 and re-removed at 17:30: `forceOn = [['2020-02-21 16:00:00', '2020-02-21 16:30:00'], ['2020-02-21 17:00:00', '2020-02-21 17:30:00']]`
# - A `directory` variable, not used in this script.

# In[1]:


## Imports // DO NOT EDIT
running_in_jupyter = True

#%matplotlib inline
import os, subprocess, shutil, configparser, datetime, sys, ast
import importlib #debug, allows to reload a module
import chromag_helper as chromag

__version__ = "v1.5.0"

if not chromag.has_screen():
    print("WARNING, this script works only if a display is connected to the session,    or if the session is open using `ssh -X` (display forwarding)")
print("Working with the version {} (commit {}), last updated on {}".format(__version__, str(chromag.get_git_revision_short_hash()), chromag.get_git_revision_date()))


# In[3]:


## ========      ===========
## Selection of the config file
## ========      ===========
## All the configuration is now in the CFG files. Here, only edit the selected folder if needed
## You can edit this if you want, overriden if running outside IPython // 
# Current options (also listed below): Array7, PFS2, 20200221, L20190410, L20190415

dataset_to_run = "L20190415"

## ==== DO NOT EDIT ==

## Load config files
config_datasets_path = "config/datasets.cfg"
config_main_path = "config/config.cfg"
assert os.path.isfile(config_datasets_path) and os.path.isfile(config_main_path)

cfg_data = configparser.ConfigParser(inline_comment_prefixes=("#",))
cfg_data.read(config_datasets_path)
cfg_main = configparser.ConfigParser(inline_comment_prefixes=("#",))
cfg_main.read(config_main_path)


print("The following datasets are available:")
for i,s in enumerate(cfg_data.sections()):
    print(" {}: {}: \t {}".format(i,s,cfg_data[s]["directory"].replace('drift-correction', 'concatenation')[1:-1]))
    
if running_in_jupyter: ## Select dataset
    pass
elif not running_in_jupyter and "--dataset-to-process" in sys.argv:
    print("(reading from command-line)")
    dataset_to_run = sys.argv[sys.argv.index("--dataset-to-process")+1]
else:
    dataset_to_run = cfg_main.get_string("Main", "dataset_to_process")
assert dataset_to_run in cfg_data.sections(), "Unrecognized dataset selected: {}, should be one of: {}".format(dataset_to_run, cfg_data.sections()) 
print("The following dataset has been selected: {}\n".format(dataset_to_run))

system = cfg_main["Main"]["system"][1:-1]
use_analysis = cfg_data[dataset_to_run]["use_analysis"][1:-1] # trim delimiter

lfn = ast.literal_eval(cfg_data[dataset_to_run]["lfn"])
forceOn = ast.literal_eval(cfg_data[dataset_to_run]["forceOn"])

## Previous analysis are saved in the file `concatenate_archive.py`
## === Do not edit
prefix_dict = ast.literal_eval(cfg_main['Main']['prefix_dict'])
fiji_dict = ast.literal_eval(cfg_main['Main']['fiji_dict'])
lfn = [i if i.endswith("/") else i+'/'  for i in lfn]
prefix = prefix_dict[system]
fiji_path = fiji_dict[system]
prefix_path = os.path.join(prefix, "data/") # Dropbox path
project_path = os.path.join(prefix, "projects/chromag") # chromag's path path
CCresult_path = os.path.join(prefix, "data/Maxime/concatenation/") # Path to store concatenated files
DCresult_path = os.path.join(prefix, "data/Maxime/drift-correction/") # Path to store the drift-corrected files

# check that files are present
assert all([os.path.isdir(chromag.convert_path(os.path.join(prefix_path, i), system)) for i in lfn]), "ERROR: some files were not found."
assert os.path.isfile(fiji_path), "ERROR: fiji not found"

# Handle date manipulations
forceLast = "2029-12-03 17:10:13"

# Sanity checks
if len(forceOn)==1 and len(forceOn[0])==1: # Make sure we have a start-stop format
    forceOn[0].append(forceLast)
assert all([len(i)==2 for i in forceOn]), "The timestamps should have the shape (t_start, t_stop)"

# Parse dates
fmt = '%Y-%m-%d %H:%M:%S'
forceOnP = [[datetime.datetime.strptime(i[0], fmt), datetime.datetime.strptime(i[1], fmt)] for i in forceOn]
assert all([i[1]>i[0] for i in forceOnP]), "End time should be after start time"
assert all([forceOnP[i[1]]<forceOnP[i[0]] for i in range(len(forceOnP)-1)]), "Not all intersections are disjunct"

## First, we will list the input files, output folders, etc.
posidict, fn = chromag.list_files_positions(prefix_path, CCresult_path, use_analysis, lfn)
print(fn)

## Log (versions, etc)
with open(os.path.join(fn, "pipeline_version.txt"), 'a') as f: ## Export pipeline version
    f.write("{}: Working with the version {} (commit {}), last updated on {}".format(datetime.datetime.now(), __version__, str(chromag.get_git_revision_short_hash()), chromag.get_git_revision_date()))

with open(os.path.join(fn, "concatenation.txt"), 'a') as f: ## Export concatenation instruction
    f.write("files:\n")
    for i in lfn:
        f.write(i+"\n")
    f.write(str(lfn))
    f.write("\nTimestamps:\n")
    f.write(str(forceOn))


# ## [1] Effectively concatenating files [independent block 1]
# Because file concatenation seems to be a mess, we decide to rely on Fiji to perform this step.
# 
# This step uses `concat.py`, a Fiji macro. /!\ Make sure you do not delete them/edit it without care :)

# In[ ]:


## Running the concatenation
## Be careful that imageJ-scifio might not read the overlay properly :s
## note that useful debug information will be displayed in the console

assert os.path.isdir(fn), "ERROR: output folder {} does not exists".format(fn)
assert chromag.has_screen(), "No display found, are you connected using `ssh -X`?"
ddf = chromag.save_timestamps(prefix_path, fn, posidict, system, fieldOn=forceOnP) ## Extract .xls spreadsheets with timestamps

tf = "timestamps.tmp"
onoff = {True: 'ON', False: 'OFF'}
for (kk,vv) in posidict.items():
    vv  = ddf[kk].path.drop_duplicates().values
    with open(tf, "w") as f:
        #[f.write(i+"\n") for i in ["{:.0f}s ({})".format(i, onoff[ii]) for (i,ii) in zip(ddf[kk].time_since_beginning.values,ddf[kk].forceActivated.values)]]
        [f.write(i+"\n") for i in ["{:.0f}s ({})".format(i, onoff[ii]) for (i,ii) in zip(ddf[kk].seconds_since_first_magnet_ON.values,ddf[kk].forceActivated.values)]]

    out_file = os.path.join(fn, 'concatenated_Pos{}.ome.tif'.format(kk))
    shutil.copyfile(tf, out_file+'.time')
    if len(vv)<=1:
        print("Copying position {}".format(kk))
        shutil.copyfile(os.path.join(prefix_path, vv[0]), out_file)
        continue
    
    if not os.path.isfile(out_file):
        print("Concatenating position {}/{}".format(kk, max(posidict.keys())))
        r = subprocess.call([fiji_path, "concat.py"]+[os.path.join(prefix_path, i) for i in vv]+[out_file, tf])
    assert os.path.isfile(out_file), "ERROR: output file was not created"
print("Done!")

## /!\ Emergency concat
emergency = False
if emergency:
    for np in [6,10,11,15,17,18,19]: # Positions to concatenate
        p = ["/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191126-before/concatenated_Pos{}.ome.tif",
             "/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191114_U2OS_stTetR-mCherry_GFP-ferritin_attraction_4bis/20191114_U2OS_stTetR-mCherry_GFP-ferritin_attraction_4_MMStack_Pos{}.ome.tif",
             "/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191126-after/concatenated_Pos{}.ome.tif"]
        out_file = "/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191126-final/concatenated_Pos{}.ome.tif".format(np)
        r = subprocess.call([fiji_path, "concat.py"]+[os.path.join(prefix_path, i.format(np)) for i in p]+[out_file, tf])

