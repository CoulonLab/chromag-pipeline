## ChroMag functions.
## By MW, Nov. 2019-March 2021
## GPLv3+

## General imports
import os, subprocess, datetime, re, pathlib, time, configparser, ast, shutil, sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict
from IPython.display import clear_output
import scipy.ndimage
from skimage.external.tifffile import TiffFile, TiffWriter
from skimage.transform import warp, EuclideanTransform
import skimage.transform


sys.path.append(os.path.join("./trackUsingMouse/"))
import tracking as trk

try:
    import matplotlib.pyplot as plt
except:
    print("Could not import matplotlib")

## ==== Wrapper functions ====
def pre_init(__version__):
    """Runs before the selection of the dataset
    __version__ (str): the version of the script
    """

    try:
        get_ipython()
        running_in_jupyter = True
    except:
        running_in_jupyter = False
    
    if not has_screen():
        print("WARNING, this script works only if a display is connected to the session,\
        or if the session is open using `ssh -X` (display forwarding)")
    print("Working with the version {} (commit {}), last updated on {}".format(__version__, str(get_git_revision_short_hash()), get_git_revision_date()))

    return running_in_jupyter

def init_config(config_paths, dataset_to_run, is_jupyter=True, verbose=False):
    """Load the config files & init the required variables

    config_paths (dict): the (relative) paths to the datasets.cfg and config.cfg files
    dataset_to_run (str): a dataset identifier matchinc one listed in datasets.cfg
    verbose (bool): whether to print the list of datasets
    """
    
    ## Sanity checks
    assert (type(config_paths) is dict) \
        and ("datasets" in config_paths) \
        and ("main" in config_paths) \
        and os.path.isfile(config_paths['datasets']) \
        and os.path.isfile(config_paths['main']) 

    ## Load two config files
    config = {}
    config['datasets'] = configparser.ConfigParser(inline_comment_prefixes=("#",))
    config['datasets'].read(config_paths['datasets'])
    config['main'] = configparser.ConfigParser(inline_comment_prefixes=("#",))
    config['main'].read(config_paths['main'])

    ## Extract which dataset to use
    if is_jupyter: ## Select dataset
        pass
    elif not is_jupyter and "--dataset-to-process" in sys.argv:
        print("(reading from command-line)")
        dataset_to_run = sys.argv[sys.argv.index("--dataset-to-process")+1]
    else:
        dataset_to_run = config['datasets'].get_string("Main", "dataset_to_process")
    
    ## Print the loaded config
    if verbose:
        print("The following datasets are available:")
        for i,s in enumerate(config['datasets'].sections()):
            print(" {}: {}: \t {}".format(i,s,config['datasets'][s]["directory"].replace('drift-correction', 'concatenation')[1:-1]))

        assert dataset_to_run in config['datasets'].sections(), "Unrecognized dataset selected: {}, should be one of: {}".format(dataset_to_run, config['datasets'].sections()) 
        print("The following dataset has been selected: {}\n".format(dataset_to_run))

    ## Extract/parse config
    config['system'] = config['main']["Main"]["system"][1:-1]
    prefix_dict = ast.literal_eval(config['main']['Main']['prefix_dict'])
    prefix = prefix_dict[config['system']]
    config['prefix'] = prefix
    
    config['lfn'] = ast.literal_eval(config['datasets'][dataset_to_run]["lfn"])
    config['lfn'] = [i if i.endswith("/") else i+'/'  for i in config['lfn']]
    config['use_analysis'] = config['datasets'][dataset_to_run]["use_analysis"][1:-1] # trim delimiter
    config['forceOn'] = ast.literal_eval(config['datasets'][dataset_to_run]["forceOn"])
    config['forceLast'] = "2029-12-03 17:10:13"
    config['ff_p'] = ast.literal_eval(config['main']["ForceField"]['ff_p'])
    config['force_field'] = config['datasets'][dataset_to_run]['force_field'][1:-1]
    if "mnp_calibration" in config['datasets'][dataset_to_run]:
        config['mnp_calibration'] = ast.literal_eval(config['datasets'][dataset_to_run]["mnp_calibration"])
    else:
        print("WARNING: `mnp_calibration` entry not found in the config file")
    
    fiji_dict = ast.literal_eval(config['main']['Main']['fiji_dict'])
    config['fiji_path'] = fiji_dict[config['system']]
    config['prefix_path'] = os.path.join(prefix, "data/") # Dropbox path
    config['CCresult_path'] = os.path.join(prefix, "data/Maxime/concatenation/") # Path
    config['DCresult_path'] = os.path.join(prefix, "data/Maxime/drift-correction/") # Path to store the drift-corrected files
    config['project_path'] = os.path.join(prefix, "projects/chromag") # chromag's path path to store concatenated files

    config['qc_mnp_file'] = config['main']["QC"]["qc_mnp_file"][1:-1]
    config['default_mnp'] = config['main']["MNP"]["default_mnp"][1:-1] 
    config['directory'] =  os.path.join(config['datasets'][dataset_to_run]["directory"][1:-1], config['use_analysis'])+"/"

    
    ## Date manipulations
    # Sanity checks
    forceOn = config['forceOn']
    if len(forceOn)==1 and len(forceOn[0])==1: # Make sure we have a start-stop format
        forceOn[0].append(config['forceLast'])
    assert all([len(i)==2 for i in forceOn]), "The timestamps should have the shape (t_start, t_stop)"

    # Parse dates
    fmt = '%Y-%m-%d %H:%M:%S'
    forceOnP = [[datetime.datetime.strptime(i[0], fmt), datetime.datetime.strptime(i[1], fmt)] for i in forceOn]
    
    assert all([i[1]>i[0] for i in forceOnP]), "End time should be after start time"
    assert all([forceOnP[i[1]]<forceOnP[i[0]] for i in range(len(forceOnP)-1)]), "Not all intersections are disjunct"
    config['forceOnP'] = forceOnP
    
    ## More sanity check
    assert os.path.isfile(config['fiji_path']), "ERROR: fiji not found"
    for i in config['lfn']:
        if not os.path.isdir(convert_path(os.path.join(config['prefix_path'], i), config['system'])):
            print("- NOT FOUND: {}".format(os.path.join(config['prefix_path'], i)))
    assert all([os.path.isdir(convert_path(os.path.join(config['prefix_path'], i), config['system'])) for i in config['lfn']]), "ERROR: some files were not found." # check that files are present
        
    return config, dataset_to_run

def init_track(fn, lfDC_all, fiji_path, cv_fn='tracking_conversion.csv', force_overwrite=False, verbose=True):
    """Check that we have everything we need to perform the tracking"""

    ## mtrk_cvfn logic
    if os.path.isfile(os.path.join(fn, cv_fn)):
        mtrk_cvfn = os.path.join(fn, cv_fn)
    else:
        mtrk_cvfn = None

    ## Extract max-intensity-projection
    lfTR_all = [i.replace(".ome.tif", "_MAX.ome.tif") for i in lfDC_all] # output path for the computed MIP

    dcl = []
    trl = []
    for i in zip(lfDC_all, lfTR_all):
        if os.path.isfile(i[1]) and not force_overwrite:
            continue
        dcl.append(i[0])
        trl.append(i[1])
    
    init_macro = "inF=newArray({});\nouF=newArray({});\n\n".format(
        ", ".join(['"{}"'.format(i) for i in dcl]),
        ", ".join(['"{}"'.format(i) for i in trl]))

    with open("./tmp/max_intensity_projection.tmp.ijm", "w") as of:
        with open("./fiji_scripts/max_intensity_projection.ijm", 'r') as f:
            of.write(init_macro)
            of.write(f.read())

    r = subprocess.call([fiji_path, "--headless", "--console", "-macro", "./tmp/max_intensity_projection.tmp.ijm"]) # can add arguments here
    
    if verbose:
        for (ii,i) in enumerate(lfTR_all): #  check what has been created
            print("{}/{}: {}: {}".format(ii+1, len(lfTR_all), os.path.basename(i),  ['NOT CREATED', 'OK'][os.path.isfile(i)]))
        print("The script 'max_intensity_projection.tmp.ijm' was created")
        print("Done!")
        
    return lfTR_all, mtrk_cvfn

def concatenate(fn, posidict, ddf, config, channels='all'):
    """Wrapper function to run the code to concatenate.
    This script prepares everything in order to runs the Fiji macro `concat.py`
    
    Parameters:
    - channels ('all' or list of int): list of channels to keep in the concatenation
    """

    tf = "./tmp/timestamps.tmp"
    onoff = {True: 'ON', False: 'OFF'}
    for (kk,vv) in posidict.items():
        vv  = ddf[kk].path.drop_duplicates().values
        with open(tf, "w") as f:
            [f.write(i+"\n") for i in ["{:.0f}s ({})".format(i, onoff[ii]) for (i,ii) in zip(ddf[kk].seconds_since_first_magnet_ON.values,ddf[kk].forceActivated.values)]]

        out_file = os.path.join(fn, 'concatenated_Pos{}.ome.tif'.format(kk))
        shutil.copyfile(tf, out_file+'.time')
        if len(vv)<=1:
            if not os.path.isfile(out_file):
                print("Copying position {}".format(kk))
                shutil.copyfile(os.path.join(config['prefix_path'], vv[0]), out_file)
            continue

        if not os.path.isfile(out_file):
            print("Concatenating position {}/{}".format(kk, max(posidict.keys())))
            chp = "--channels={}".format(str(channels).replace(' ', ''))
            r = subprocess.call([config['fiji_path'], "./fiji_scripts/concat.py"]+[os.path.join(config['prefix_path'], i) for i in vv]+[out_file, tf, chp])
        assert os.path.isfile(out_file), "ERROR: output file was not created"
    print("Done!")    

def bandpass(fn, posidict, config):
    """Wrapper function to run the code to compute the bandpassed version of the movies
    This script prepares everything in order to runs the Fiji macro `process.py`
    """
    for (kk,vv) in posidict.items():
        out_file = os.path.join(fn, 'concatenated_Pos{}.ome.tif'.format(kk))
        out_fileBP = os.path.join(fn, 'concatenated_Pos{}BP.ome.tif'.format(kk))
        if not os.path.isfile(out_file+'.txt'):
            print("File {} will not be included in further analysis".format(out_file))
        elif os.path.isfile(out_file+'.txt') and os.path.isfile(out_fileBP):
            print("Skipping {}, file already exists".format(os.path.basename(out_fileBP)))
        if os.path.isfile(out_file+'.txt') and not os.path.isfile(out_fileBP):
            with open(out_file+'.txt', 'r') as f:
                ll = f.readlines()
                s1 = [l for l in ll if l.startswith('sd1')]
                s2 = [l for l in ll if l.startswith('sd2')]
            sds = [("{}, {}".format(i[0],i[1])).replace('\n', '') for i in list(zip(s1,s2))]
            assert all([i==sds[0] for i in sds]), (out_file, sds)
            print("Computing bandpassed for {}, with parameters {}".format(os.path.basename(out_file), sds[0]))
            r = subprocess.call([config['fiji_path'], "-Xmx40G", "--", "--jython", "./fiji_scripts/process.py", out_file, out_fileBP, "0", "add_bp"]) # can add arguments here
            assert os.path.isfile(out_fileBP), "The bandpassed file was not created for file {}".format(out_fileBP)
    print("Done!")    

def crop(fn, config, correction_type="none", followLocus=False, verbose=True):
    """Wrapper function to run the code to crop individual cells.
    This script prepares everything in order to runs the Fiji macro `process.py`

    **Note:** this step can also be used to perform a drift-correction step, 
    as it was documented in a previous version of the pipeline. 
    This option is now deprecated.

    correction_type (str): [followLocus, followSirDNA, none]
    """

    ## Arguments logic & sanity checks
    assert has_screen(), "No display found, are you connected using `ssh -X`?"
    if correction_type == "followLocus":
        followLocus = True # Set to True to register on the locus channel (not sirDNA). // so far, keep it to False!

    ## Create folder architecture
    lf = [os.path.join(fn, i) for i in os.listdir(fn) if i.endswith('.ome.tif') and has_dc_instructions(os.path.join(fn, i))]

    lfDC = []
    for i in lf: # create folder architecture
        p = i.replace(config['CCresult_path'], config['DCresult_path'])
        assert p!=i, "The input file is the same as the output file. ABORTING"
        if not os.path.isdir(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p))
        lfDC.append(p)

    assert all([has_dc_instructions(i) for i in lf]), "Some file are missing Drift Correction instructions. Please fix this."

    if verbose:
        print("Found {} files".format(len(lf)))

    ## Perform crop (previously: drift correction)
    lfDC_all = []
    for (ii,  i) in enumerate(zip(lf, lfDC)):
        p,pDC = i
        for c in range(get_number_cells(p)):
            if verbose:
                print("{}/{} Processing: {} (cell {})".format(ii+1, len(lf), os.path.basename(p), c))
                #clear_output(wait=True)        
            of = pDC.replace('.ome.', 'DC'+str(c)+'.ome.')
            if followLocus:
                of = pDC.replace('.ome.', 'DClocus'+str(c)+'.ome.')                
            lfDC_all.append(of)
            if os.path.isfile(of):
                continue
            r = subprocess.call([config['fiji_path'], "--jython", "./fiji_scripts/process.py", p, of, str(c), str(correction_type)]) # can add arguments here
            assert os.path.isfile(of) and (correction_type=='none' or os.path.isfile(of+'.shifts')), "File not created, aborting!"        
        shutil.copyfile(p+'.txt', pDC+'.txt')
    if verbose:
        print("Done!")        
        
    return lf, lfDC, lfDC_all

def track():
    """Should be a wrapper for trackUsingMouse"""

def track_subpixel(fn, lfTR_all, lfDC_all, mtrk_cvfn, select="", skip=-1, border=3, dbg=False, verbose=True):
    """We implement a 2D+1D fitting, a version that should also work in the edges
    There is an "offsets" parameters that can be tuned to recover improperly drift-corrected folders.
    
    Options:
    - select (str): set to "19DC0" to only process the corresponding file, or to None/empty string instead
    - skip (int): set to n>=0 to skip the n first files
    - dbg (bool): run in debug mode (not sure what it does)
    - verbose (bool): print many things on the screen"""
    
    shifts_prefix = os.path.join(Path(lfDC_all[0]).parent.parent, "shifts")
    if os.path.isdir(shifts_prefix): # Offset for improperly inputted coordinates. We see if a "shifts" folder exists 
        use_offsets=True
        lfSH_all = [os.path.join(shifts_prefix, os.path.basename(i)+".shifts") for i in lfDC_all] # a folder to store the right shifts
    else:
        use_offsets=False

    summary = []
    for (ii,k) in enumerate(zip(lfDC_all, lfTR_all)):
        i,j=k
        # Manage skipping/debugging
        if skip>0:
            skip-=1
            continue
        if select is not None and select not in i:
            print("skipping, we want to process {}".format(select))
            continue

        print("{}/{}: Fitting: {}".format(ii+1, len(lfDC_all), os.path.basename(i)))
        mtrk = i.replace(".ome.tif", ".ome_.mtrk")
        if use_offsets and os.path.isfile(lfSH_all[ii]):
            offsets = pd.read_csv(lfSH_all[ii])
        elif use_offsets and not os.path.isfile(lfSH_all[ii]):
            print("Offset file not found: {}.".format(lfSH_all[ii]))
            offsets = None
        else:
            offsets = None

        fitme = False
        if os.path.isfile(mtrk): # tracking exists
            chLocus = 1 # random
            fitme=True
        elif (not os.path.isfile(mtrk)) and (mtrk_cvfn is not None) and os.path.isfile(os.path.join(fn, mtrk_cvfn)):
            fitme=True

            # read the config file & extract xLoc, yLoc
            bn = os.path.basename(mtrk).split('DC')[0]+'.ome.tif' # the concatenated file
            cfg_p = os.path.join(fn, bn)
            cfg = load_config_file(cfg_p)
            c = int(os.path.basename(mtrk).split('DC')[1].split('.ome')[0])
            ns = cfg.sections()[c] # name section

            # we need to offset the ROI coordinates if we provide a cv_fn , so we set fix_roi to True
            mtrk, offsets = load_mtrk(fn, img_p=i, mtrk_cvfn=mtrk_cvfn, fix_roi=True, cfg_p=cfg_p, cfg_c=c)

            # extract chLocus
            if cfg.getint(ns, 'chDNA')==1:
                chLocus=2
            else:
                chLocus=1

        if fitme:
            # set disp=3 to debug
            # with PsfZpx==40, we perform a global fit in the z dimension
            #from IPython.core.debugger import set_trace; set_trace()
            trk.trackUsingMouse(j, '_', mtrk_file=mtrk, psfPx=4.0, psfZPx=7.0, offsets=offsets, thresholdSD=5,
                                force2D=True, extractZ=True, channel2D=chLocus, fnTif3D=i, fps=-1, border=border, disp=2, verbose=verbose)
            # old psfPx = 7, old thresholdSD=5 (default)
            da = pd.read_csv(mtrk.replace(".mtrk", ".trk2"), sep='\t')
            n_cvg = (da['Code (bits: spot detected, fit convereged)']>=2).sum() # keep the typo!
            txt = "[Fitting done] {} ({}/{} fits converged)".format(os.path.basename(i), n_cvg, da.shape[0])
            summary.append(txt)        
        else:
            txt = "[No tracking data] {}".format(os.path.basename(i))
            summary.append(txt)
            if verbose:
                print(txt)
        if skip==0:
            break

    print("Summary of fitting:")
    for (kii,ki) in enumerate(summary):
        print("{}: {}".format(kii+1,ki))
    print("Done!")
    return mtrk_cvfn

def track_debug(pat, lfTR_all, lfDC_all, ch=1):
    """
    pat: a pattern to filter which cell to load, e.g: '22DC3'
    ch: locus channel"""
    dbg_ch = ch
    dbg_imgp = [i for i in lfTR_all if pat in i][0]
    dbg_trkp = [i for i in lfDC_all if pat in i][0].replace('.ome.tif', '.ome_.trk2')
    assert os.path.isfile(dbg_imgp) and os.path.isfile(dbg_trkp)
    dbg_trk = pd.read_csv(dbg_trkp, sep='\t')
    dbg_img = TiffFile(dbg_imgp).asarray() # t, ch, x, y: it is the max-intensity-projected file


    c={3: 'red', 1: 'blue', 0: 'orange', 2: 'pink'}
    print("Color code:")
    for (k,v) in c.items():
        print(f"code {k}: {v}")
    
    plt.figure(figsize=(18, 40))
    for i in range(dbg_img.shape[0]):
        l=dbg_trk.loc[dbg_trk['# Frame']==i]
        x=l['Position X']
        y=l['Position Y']
        st=l['Code (bits: spot detected, fit convereged)']
        plt.subplot(dbg_img.shape[0]//3+1,3,i+1)
        try:
            plt.scatter(x.values[0],y.values[0], color=c[st.values[0]])
        except:
            from IPython.core.debugger import set_trace; set_trace()
        plt.imshow(dbg_img[i, dbg_ch], cmap='gray')
        plt.title("Frame {}: x={}, y={}".format(i, x.values[0],y.values[0]))
    
    return dbg_trk

def forcefield(fn, ff_p, project_path, prefix_path, force_name,
               tabname="small array", plot=True, verbose=True):
    """Wrapper code to load a forcefield"""
    
    ## Load force field
    if verbose:
        print("Loading force field: {}".format(force_name))
    imgF, r_force, px_size, simulated_pillar_pointing_down = load_forcefield(prefix_path, ff_p, force_name) ## Load and scale the force field.

    if plot:        
        plt.figure(figsize=(18,8))
        plt.imshow(imgF[0], cmap='BrBG')
        plt.plot([i[0] for i in r_force], [i[1] for i in r_force])
        plt.colorbar()
        plt.xlim(0,imgF.shape[1])
        plt.ylim(imgF.shape[2], 0)
        
    return imgF, r_force, px_size, simulated_pillar_pointing_down

def list_Fstart(lf, lfDC, imgF, px_size, r_force, config, verbose=True):
    r = []
    for (i, ii) in enumerate(zip(lf, lfDC)):
        p,pDC = ii
        nc = get_number_cells(p)
        for c in range(nc):
            if verbose:
                print("File {}, cell {}".format(os.path.basename(p), c))
            rr = get_Fstart(p, pDC, c, imgF, px_size, r_force, config)
            r.append({'movie': os.path.basename(p), 'cell': c,
                      'Fstart_per_MNP_fN': rr[1][0],
                      'Fstart_per_locus_pN': rr[3],
                      'MNPs': rr[2],
                      'Fx_start_per_MNP_fN': rr[0][0][0],
                      'Fy_start_per_MNP_fN': rr[0][1][0],
                      'Fz_start_per_MNP_fN': rr[0][2][0]})
    return pd.DataFrame(r)

def export_csv(fn, lf, lfDC, imgF, r_force, px_size,
               config, ignore=[], ignore_missing=False, select="", scl=10000, plot=True, export_pdf=True):
    """A wrapper version to export the first version of the CSV. This function does:
    1. extract the force value at the locus
    2. Extract the number of particles at the locus
    3. (optional) plot the number of particles at the locus
    
    Parameters:
    - ignore (list of tuples)#, [(1,0)] #3DC0, need to be added for Array7
    - ignore_missing (bool): ignores all files that are missing a .mtrk/.trk2 file.
    - select (str): a pattern to select (e.g: "Pos11_cell0")
    """

    ## unwrap config
    default_mnp = config['default_mnp']
    simulated_pillar_pointing_down = config['simulated_pillar_pointing_down']
    mtrk_cvfn = config['mtrk_cvfn']
    qc_mnp_file = config['qc_mnp_file']
    prefix = config['prefix']
    
    force_overwrite = True # TODO MW: to move elsewhere

    ## Load the QC file
    ncells = sum([get_number_cells(i) for i in lf])
    qc_mnp = pd.read_excel(os.path.join(prefix, qc_mnp_file))
    qc_mnp.dropna(thresh=5, inplace=True) # Remove Antoine's comments :)
    qc_p = [os.path.join(prefix, pa, fi) for (pa,fi) in zip(qc_mnp.path, qc_mnp.file)]
    qc_mnp['p']=qc_p

    if plot:
        plt.figure(figsize=(13,5*ncells))
        counter = 0
        allPlots = []
        plot_intensities = [] # the data needed to plot the intensities over time
    
    for (i,tmp) in enumerate(zip(lf,lfDC)):
        (p,pDC) = tmp
        c_nb = get_number_cells(p)
        for c in range(c_nb):
            print("{}/{} -- Loading cell {}/{} ({})".format(i+1, len(lf), c, c_nb, os.path.basename(p)))
            #clear_output(wait=True)
            if (i,c) in ignore:
                print("WARNING: File {} cell {} ignored".format(p,c))
                continue
            if select != "" and select not in pDC.replace('.ome.tif', '_cell{}.csv'.format(c)):
                continue

            ## Load the cell
            res = load_cell(fn, p, pDC, c, track_type='trackusingmouse', verbose=False, 
                            apply_shifts=False, load_bp=True, mtrk_cvfn=mtrk_cvfn)
            if res['flag']!='ok':
                print("Cell not loaded, skipping. Reason: {}".format(res['flag']))
                continue
            
            img = res['img']
            tr_dr = res['tr_dr']
            rect = res['rect']
            rect2 = res['rect2']
            pillar = res['pillar']
            zLoc = res['zLoc']
            chDNA = res['chDNA']
            pixXY = res['pixXY']
            assert (tr_dr.t.max()+1) >= img.shape[0], "missing tracking data (image data has {} frames, whereas the data has {} lines)".format(img.shape[0], tr_dr.t.max())

            ## Map the force & compute averages/medians
            FM, (Fx, Fy, Fz) = map_force(imgF, tr_dr.x_DC, tr_dr.y_DC, px_size, pixXY, r_force, pillar, simulated_pillar_pointing_down)

            tr_dr['Fx']=Fx
            tr_dr['Fy']=Fy
            tr_dr['Fz']=Fz
            
            tr_dr['dx'] = -tr_dr.x.diff(-1) # The displacement (after drift correction)
            tr_dr['dy'] = -tr_dr.y.diff(-1) # We put -1 so that the dx at time t is the displacement to be made and not the past one
            Fnn = (tr_dr.Fx**2+tr_dr.Fy**2+tr_dr.Fz**2)**0.5
            Fxm = (tr_dr.Fx/Fnn).median(skipna=True)
            Fym = (tr_dr.Fy/Fnn).median(skipna=True)
            Fzm = (tr_dr.Fz/Fnn).median(skipna=True)
            Fnm = (Fxm**2+Fym**2+Fzm**2)**0.5 # norm of the force

            tr_dr['Fx_m'] = Fxm/Fnm*Fnn
            tr_dr['Fy_m'] = Fym/Fnm*Fnn
            tr_dr['Fz_m'] = Fzm/Fnm*Fnn

            ## Convert the force per particle to a total force
            cal_i = config['mnp_calibration']
            calp = os.path.join(os.path.dirname(p), cal_i['movie'].split('DC')[0]+'.ome.tif')
            calpDC = os.path.join(os.path.dirname(pDC), cal_i['movie'].split('DC')[0]+'.ome.tif')
            calc = int(cal_i['movie'].split('DC')[1].replace('.ome.tif', ''))
            tr_dr_calp = os.path.join(os.path.dirname(pDC), cal_i['movie'].replace('DC', '_cell').replace('.ome.tif', '.csv'))
            tr_tmp = load_cell(fn, calp, calpDC, calc, track_type='trackusingmouse',
                              apply_shifts=False, trdr_only=True,
                              mtrk_cvfn=mtrk_cvfn)
            if tr_tmp['flag'] != 'ok':
                raise Exception(tr_tmp['flag'])
            else:
                tr_dr_cal = tr_tmp['tr_dr']
            its = tr_dr_cal.loc[tr_dr_cal.t==int(cal_i['frame'])]['Fluo. instensity'].values[0]

            if 'intensity_per_particle' in tr_dr.columns:
                del tr_dr['intensity_per_particle']
            tr_dr['intensity_per_particle'] = its/float(cal_i['MNPs'])
            cv_bits = tr_dr['Code (bits: spot detected, fit convereged)']
            mnp_auto = (tr_dr.loc[(tr_dr.t>0)&(tr_dr.t<=5)&(cv_bits>=2)]['Fluo. instensity']/
                   tr_dr.loc[(tr_dr.t>0)&(tr_dr.t<=5)&(cv_bits>=2)]['intensity_per_particle']).median()
            tr_dr['MNP_auto'] = mnp_auto
            
            cP = pDC.replace(".ome.tif", "DC{}.ome.tif".format(c))
            tr_dr['MNP_manual'] = None
            if cP in qc_mnp.p.values: # Make sure that we have MNP/QC data for this dataset
                tr_dr['MNP_manual'] = qc_mnp[qc_mnp.p==cP].locus_MNP.values[0]
            tr_dr['MNPs'] = tr_dr[default_mnp]

            ## Then we make the plots
            if plot:
                (forX,forY,forZ)=FM
                plot_intensities.append([p,c,tr_dr['t'], tr_dr['Fluo. instensity']])
            
                x_filt = tr_dr.x_DC[tr_dr.x>10]
                y_filt = tr_dr.y_DC[tr_dr.x>10]

                plt.subplot(ncells, 2, 2*counter+1)
                plt.imshow(img[2,int(tr_dr.z[2]),1])
                plt.plot(rect[0], rect[1], color='red')
                plt.plot([i[0] for i in pillar], [i[1] for i in pillar])
                plt.plot(x_filt, y_filt, color='orange')
                plt.xlim(0,img.shape[-2])
                plt.ylim(0,img.shape[-1])
                plt.title("{} - cell {}".format(os.path.basename(p), c))

                plt.subplot(ncells, 2, 2*counter+2)
                plt.imshow(masks_to_outlines(np.abs(forX)>1e-10, usena=True), cmap='gray_r')
                plt.imshow(forX, alpha=0.9, cmap='BrBG')
                #plt.plot([i[0] for i in pillar], [i[1] for i in pillar])
                plt.plot([pillar[-1][0]]+[i[0] for i in pillar], [pillar[-1][1]]+[i[1] for i in pillar])
                plt.colorbar()
                plt.plot(rect[0], rect[1], color='red')
                plt.plot(x_filt, y_filt, color='orange')
                plt.xlim(0,img.shape[-2])
                plt.ylim(0,img.shape[-1])
                plt.arrow(tr_dr.x_DC[1], tr_dr.y_DC[1], tr_dr.Fx[1]*scl, tr_dr.Fy[1]*scl, width=1, color='red')
                counter +=1
                #chromag.plot_registration(res['pillar'], forX, forY, forZ, img, zLoc, chDNA, tr_dr)
                
            if ('dx_R' in tr_dr) or ('dy_R' in tr_dr):
                print("Replacing `dx_R` and `dy_R` columns")
                del tr_dr['dy_R']
                del tr_dr['dx_R']
                del tr_dr['Fx_R']
                del tr_dr['Fy_R']

            tr_dr.to_csv(pDC.replace('.ome.tif', '_cell{}.csv'.format(c)))
            clear_output(wait=True)
            if select != "" and select in pDC.replace('.ome.tif', '_cell{}.csv'.format(c)):
                raise IOError("We break here")
    
    if plot and export_pdf:
        plt.savefig(os.path.join(fn, "pillar_registration.pdf"))
        
    if plot:
        plt.figure(figsize=(18,6))
        for (i_n, i_c, i_t, i_d) in plot_intensities:
            plt.plot(i_t, i_d, label="{} cell{}".format(os.path.basename(i_n).split("Pos")[-1].split(".")[0], i_c))
        plt.legend()
        plt.xlabel("time (frame)")
        plt.ylabel("locus intensity (AU)")
        if export_pdf:
            plt.savefig(os.path.join(fn, "locus_intensities.pdf"))

    print("-- All done!")

def map_force(imgFF, x, y, pxFF, px, r_force, pillar, simulated_pillar_pointing_down):
    """Compute the mapping of the force field onto the movie.
    Function wraps the following steps:
    - pillar registration (xy scaling + registration)
    - mapping of the input coordinates
    
    Inputs:
    - imgFF (array): array representing the simulated force field around pillar
    - x, y (1D arrays): the x and y coordinates where the force should be mapped
    - pxFF (float): pixel size of the simulated force field (imgFF)
    - px (float): pixel size of the movie
    
    - simulated_pillar_pointing_down (bool): whether the pillar is pointing down in imgFF
    
    Outputs:
    - MappedForceField (tuple of 2D arrays; (forX, forY, forZ)): the mapped force field, each array is one force coordinate
    - MappedForce (tuple of 1D arrays; (fx, fy, fz)): the force coordinates at the input points (x,y,z)
    """
      
    # For memory:
    #fx = tr_dr['Fx']
    #x = tr_dr.x_DC
    #y = tr_dr.y_DC
    
    ## Map the force field
    imgFR, r_forceR = rescale_field(imgFF, pxFF, px, r_force) ## Here we should rescale the force field.
    
    if np.all(np.array([i[1] for i in pillar])>0):
        loc='bottom'
    else:
        loc='top'
    
    forX, Tx = get_registered_pillar(r_forceR, pillar, imgFR[0], pillar_location=loc, out_shape='default', debug=False, flip_force_field=simulated_pillar_pointing_down)
    forY, Ty = get_registered_pillar(r_forceR, pillar, imgFR[1], pillar_location=loc, out_shape='default', flip_force_field=simulated_pillar_pointing_down)
    forZ, Tz = get_registered_pillar(r_forceR, pillar, imgFR[2], pillar_location=loc, out_shape='default', flip_force_field=simulated_pillar_pointing_down)

    if loc=='top':
        forY=-forY
    if loc=='bottom':
        forX=-forX

    ## Extract the force at the location of the locus
    fx = scipy.ndimage.map_coordinates(forX, np.vstack((y,x)))
    fy = scipy.ndimage.map_coordinates(forY, np.vstack((y,x)))
    fz = scipy.ndimage.map_coordinates(forZ, np.vstack((y,x)))
        
    return (forX,forY,forZ), (fx, fy, fz)

    
def export_csv2(config_segment, select="", select_f=[],
                export_rot=False, export_rot_fullsize=True, verbose=True):
    """
    Wrapper function to export _2.csv files and _rot files.

    Parameters:
    - select (str, default: ''): a string pattern used to only process some files. 
      For instance 'Pos0DC0'.
    - select_f (list of ints): frame(s) to select, a list, -1 or empty list to 
      process everything
    """

    ## Unwrap config
    img_p = config_segment['img_p']
    prefix = config_segment['prefix']
    directory = config_segment['directory']
    trk_p = config_segment['trk_p']
    csv_p = config_segment['csv_p']

    def print_(x, verbose, **kwargs):
        if verbose:
            print(x, **kwargs)
    
    for i in range(len(img_p)):
        if select not in img_p[i]:
            continue
        print_("processing {}/{}".format(i+1, len(img_p)), verbose)
        tmp = compute_metrics(prefix, directory, index=i, 
                                      img_p=img_p, trk_p=trk_p, csv_p=csv_p, scl=500,
                                      select_f=select_f,
                                      verbose_ch=2, verbose=False, verbose_sbs=False)
        if tmp['err'] != "":
            print_("ERROR: an error happened: {}".format(tmp['err']), verbose)
            continue

    print_("All done!", verbose)

def bleach_correction(config_segment, plot=True, save_plot=True, verbose=True):
    """Extract values required to normalize the intensity of the movie over time"""
    
    ## Unwrap config
    img_p = config_segment['img_p']
    trk_p = config_segment['trk_p']
    csv_p = config_segment['csv_p']
    prefix = config_segment['prefix']
    directory = config_segment['directory']

    def print_(x, verbose, **kwargs):
        if verbose:
            print(x, **kwargs)    
    df = []

    for i in range(len(img_p)):
        print_("processing {}/{}".format(i+1, len(img_p)), verbose)
        im, im1, da, tr, ex, ip, cp, ap = read_data(prefix, directory, i, img_p, trk_p, csv_p) # Load data for the rest of the analysis
        z_dict = {i.t:int(i.z)-1 for (_,i) in da.iterrows()} # Extract Z position || #np.all(im[f,z_dict[f],2]==im1[f,2]) # sanity check
        im2=np.asarray([im[f,max(0,z_dict[f]-1):min(im.shape[1], z_dict[f]+2)].mean(axis=0) for f in range(im1.shape[0])]) # We average over 3 consecutive z planes
        if not os.path.isdir(ap):
            os.makedirs(ap)
        apNORM = os.path.join(ap,'norm/')
        if not os.path.isdir(apNORM):
            os.makedirs(apNORM)
        if plot:
            plt.figure(figsize=(18,18))
        for f in range(im1.shape[0]):
            mask = im1[f,3]
            P = da[da.t==f]

            # Determine which cell index to use
            x=int(round(P.x.values[0]))
            y=int(round(P.y.values[0]))
            cell_index=-1
            if mask.max()==0 or x<=10 or y<=10:
                print_("No segmentation found on frame {}".format(f), verbose)
                df.append({'cell': i, 'frame': f, 'file': img_p[i]})
                continue        
            if mask[y,x]!=0:
                cell_index=mask[y,x]
            if cell_index==-1:
                print_("Cell with locus not found on frame {}, skipping".format(f), verbose)
                df.append({'cell': i, 'frame': f, 'file': img_p[i]})
                continue

            pxs = im1[f,2][mask==cell_index] # the corresponding pixels
            pxs2 = im2[f,2][mask==cell_index] 
            df.append({'cell': i, 'frame': f,  'file': img_p[i], 
                       'area': len(pxs), 'x':x, 'y':y,
                      'mean': pxs.mean(), 'std': pxs.std(), 
                      'mean_avg': pxs2.mean(), 'std_avg': pxs2.std()})

            if plot:
                plt.subplot(im1.shape[0]//3+1, 3, f+1)
                plt.hist(pxs, bins=np.linspace(0, pxs.max(), num=100))
                plt.title("Frame {}".format(f))
        if save_plot:
            plt.savefig(os.path.join(apNORM, "intensity_hist.pdf"))
        if plot:
            plt.tight_layout()
    print_("All done!", verbose)

    ## == Pretty plots
    apBLEACH = os.path.join(prefix, directory)
    dff = pd.DataFrame(df)

    var=["mean", "mean_avg", "std", "std_avg"]
    dff_m = dff.groupby('frame').mean()[var]
    dff_s = dff.groupby('frame').std()[var]
    dff_m.columns=[i+'_mean' for i in dff_m.columns]
    dff_s.columns=[i+'_std' for i in dff_s.columns]
    dff_ = pd.merge(dff_m, dff_s, on='frame')

    dff_t = dff_.copy()
    dff_t.columns = ["ALLCELLS_"+i for i in dff_t.columns]
    dff = dff.merge(dff_t, how='left',  on="frame")
    dff['ALLCELLS_mean_norm'] = dff.ALLCELLS_mean_mean.max()/dff.ALLCELLS_mean_mean
    dff['ALLCELLS_mean_avg_norm'] = dff.ALLCELLS_mean_avg_mean.max()/dff.ALLCELLS_mean_avg_mean
    dff['ALLCELLS_std_norm'] = dff.ALLCELLS_std_mean.max()/dff.ALLCELLS_std_mean
    dff['ALLCELLS_std_avg_norm'] = dff.ALLCELLS_std_avg_mean.max()/dff.ALLCELLS_std_avg_mean
    dff.to_csv(os.path.join(apBLEACH, "bleaching.csv"))

    if plot:  ## Plot bleaching curves
        plt.figure(figsize=(18,16))

        plt.subplot(321)
        for (g,d) in dff.groupby('cell'):
            plt.plot(d.frame,d['mean'], label=g)

        plt.ylim(0,dff['mean'].max()+100)
        plt.title("Mean nuclear fluorescence (single plane)")
        plt.xlabel("Time (frames)")

        plt.subplot(322)
        for (g,d) in dff.groupby('cell'):
            plt.plot(d.frame,d['std'], label="{}".format(img_p[int(g)][16:].split('.')[0]))
        plt.ylim(0,dff['std'].max()+100)
        plt.title("Standard deviation nuclear fluorescence (single plane)")
        plt.legend()
        plt.tight_layout()
        plt.xlabel("Time (frames)")

        plt.subplot(323)
        for (g,d) in dff.groupby('cell'):
            plt.plot(d.frame,d['mean_avg'], label=g)

        plt.ylim(0,dff['mean_avg'].max()+100)
        plt.title("Mean nuclear fluorescence (3-planes average)")
        plt.xlabel("Time (frames)")

        plt.subplot(324)
        for (g,d) in dff.groupby('cell'):
            plt.plot(d.frame,d['std_avg'], label="{}".format(img_p[int(g)][16:].split('.')[0]))
        plt.ylim(0,dff['std_avg'].max()+100)
        plt.title("Standard deviation nuclear fluorescence (3-planes average)")
        plt.legend()
        plt.tight_layout()
        plt.xlabel("Time (frames)")

        plt.subplot(325)
        varI = var[1] #mean_avg
        plt.plot(dff_.index, dff_[varI+"_mean"], linewidth=2)
        plt.plot(dff_.index, dff_[varI+"_mean"]-dff_[varI+'_std'], color='gray', linestyle="--")
        plt.plot(dff_.index, dff_[varI+"_mean"]+dff_[varI+'_std'], color='gray', linestyle="--")
        plt.title(varI)
        plt.ylim(0,None)
        plt.xlabel("time (frame)")
        plt.ylabel("Average intensity (AU)")

        plt.subplot(326)
        varI = var[3] #std_avg
        plt.plot(dff_.index, dff_[varI+"_mean"], linewidth=2)
        plt.title(varI)
        plt.ylim(0,None)
        plt.xlabel("time (frame)")
        plt.ylabel("Average intensity (AU)")

        plt.savefig(os.path.join(apBLEACH, "bleaching.pdf"))

    # save documentation
    doc = {'area': "Number of pixels lying in the 2D mask", 
           'cell': "index of the file (in `img_p`)",
           'file': "filename",
           'frame': "frame number (0-indexed)",
           'mean': 'mean intensity value (averaged over the 2D mask and 1 z-plane)',
           'std': 'standard deviation of the intensity value (averaged over the 2D mask and 1 z-plane)',
           'mean_avg' : 'mean intensity value (averaged over the 2D mask and 3 z-plane)',
           'std_avg' : 'standard deviation of the intensity value (averaged over the 2D mask and 3 z-plane)',
           'x': 'x coordinate of the locus (in pixels)',
           'y': 'y coordinate of the locus (in pixels)',
           'ALLCELLS_mean_mean': "column `mean` averaged for all cells (computed per frame)",
           'ALLCELLS_mean_avg_mean': "column `mean_avg` averaged for all cells (computed per frame)",
           'ALLCELLS_std_mean': "column `std` averaged for all cells (computed per frame)",
           'ALLCELLS_std_avg_mean': "column `std_avg` averaged for all cells (computed per frame)",
           'ALLCELLS_mean_std': "standard deviation of the column `mean` (computed per frame)",
           'ALLCELLS_mean_avg_std': "standard deviation of the column `mean_avg` (computed per frame)",
           'ALLCELLS_std_std': "standard deviation of the column `std` (computed per frame)",
           'ALLCELLS_std_avg_std': "standard deviation of the column `std_avg` (computed per frame)",
           'ALLCELLS_mean_norm': "per-frame normalization constant, based on the `mean` intensity value",
           'ALLCELLS_mean_avg_norm': "per-frame normalization constant, based on the `mean_avg` intensity value",
           'ALLCELLS_std_norm': "per-frame normalization constant, based on the `std` intensity value",           
           'ALLCELLS_std_avg_norm': "per-frame normalization constant, based on the `std_avg` intensity value",                  
    }
    with open(os.path.join(apBLEACH, "bleachingDOC.txt"), "w") as f:
        f.write("Documentation of the file `bleaching.csv`\n")
        for k,v in doc.items():
            f.write("{}: {}\n".format(k,v))
    return dff

def export_csv3(fn, dff, qc_mnp_file, config_segment, plot=True, verbose=True):
    """Merge all the data we have in one _3.csv file per cell"""
    
    ## Unwrap config
    csv_p = config_segment['csv_p']
    prefix = config_segment['prefix']
    directory = config_segment['directory']

    def print_(x, verbose, **kwargs):
        if verbose:
            print(x, **kwargs)    

    print_("Working in directory {}".format(os.path.join(prefix, directory)), verbose)
    csv2_p = []
    for i in csv_p:
        if not os.path.isfile(os.path.join(prefix, directory, i.replace('.csv', '_2.csv'))):
            print_("File {} was not processed".format(i), verbose)
        else:
            csv2_p.append(i.replace('.csv', '_2.csv'))

    print_("\nWorking with the following files:", verbose)

    # list all files
    for i,fn in enumerate(csv2_p):
        tr = pd.read_csv(os.path.join(prefix, directory, fn))
        print("-- {}/{}: {} has {}/{} segmented frames".format(i+1, len(csv2_p), fn, tr.Df.notna().sum(), tr.shape[0]))
        plt.plot(tr.t.values, tr.Dmb.values)
    plt.xlabel("time (s)")
    plt.ylabel("Distance to the membrane")

    ## Load QC data
    qc_mnp = pd.read_excel(os.path.join(prefix, qc_mnp_file))
    qc_mnp.dropna(thresh=5, inplace=True) # Remove Antoine's comments :)
    qc_p = [os.path.join(prefix, pa, fi) for (pa,fi) in zip(qc_mnp.path, qc_mnp.file)]
    qc_mnp['p']=qc_p
    qc_mnp_framebyframe = qc_mnp.copy()
    for n in qc_mnp.columns:
        if type(n) is str and n.startswith("Unnamed: "):
            del qc_mnp[n]
            del qc_mnp_framebyframe[n]
        elif type(n) is int: # remove frame information in qc_mnp, but not 
            del qc_mnp[n]
        elif n not in ('path', 'p'):
            del qc_mnp_framebyframe[n]        
    qc_mnp.columns = ["qc_{}".format(i) for i in qc_mnp.columns]
    qc_mnp_framebyframe.columns = ["qc_{}".format(i) for i in qc_mnp_framebyframe.columns]

    qc_m = qc_mnp_framebyframe.melt(id_vars=['qc_path', 'qc_p'])
    qc_m.columns = ['qc_path', 'qc_p', 'frame', 'qc_flags']
    qc_m.frame = qc_m.frame.str.replace('qc_', '').astype(int)-1 # convert to 0-indexed frame number

    # plot distance as a function of force, motion, etc.
    # Make plots and plot the side-by-side
    xls_p = [i for i in os.listdir(os.path.join(prefix, directory.replace("drift-correction", "concatenation"))) if i.endswith('.xls')]

    d1 = {i.split('_')[1].replace('.xls',''): [i] for i in xls_p}
    for i in csv2_p:
        d1[i.split('_')[1]].append(i)

    for i in d1.keys():
        xl = pd.read_excel(os.path.join(prefix, directory.replace("drift-correction", "concatenation"), d1[i][0]))
        for j in d1[i][1:]:
            tr = pd.read_csv(os.path.join(prefix, directory, j))
            key = j.replace("_cell", "DC").replace("_2.csv", ".ome.tif")
            dff_tmp = dff[dff.file==key]
            if not dff_tmp.shape[0]>1:
                print("File {} does not seem to be processed".format(key))
                continue

            tr['frame1']=tr.frame+1
            xlm = xl.merge(tr, left_on='frame', right_on='frame1', how='outer')
            xlm = xlm.merge(dff_tmp, left_on='frame_y', right_on='frame', suffixes=('', '_INT'))

            # Add QC data
            xlm_p = os.path.join(prefix, directory, j).replace("_cell", "DC").replace("_2.csv", ".ome.tif")
            if "qc_p" in xlm.columns:
                del xlm['qc_p']
            xlm["qc_p"]=xlm_p
            xlm = pd.merge(xlm, qc_mnp, how='left', on='qc_p')
            xlm = xlm.merge(qc_m, how='left', on=('qc_p', 'frame'))

            for n in xlm.columns:
                if n.startswith('Unnamed: '):
                    del xlm[n]

            ## Compute angle at t=0s (should not be here...)
            #print(t.columns, tr.columns, da.columns)
            ts_s = xlm.seconds_since_first_magnet_ON
            t0 = xlm.frame[ts_s <= 0].max()-1 # 0-indexed
            Fx0 = xlm.Fx[xlm.frame==t0].values[0]
            Fy0 = xlm.Fy[xlm.frame==t0].values[0]
            angle0s = angle_between(np.array([Fx0, Fy0]), np.array([0,-1]))/np.pi*180
            xlm['angle0s']=angle0s
            
            ## Save
            xlm.to_csv(os.path.join(prefix, directory, j.replace("_2.csv", "_3.csv"))) # save _3.csv

            ## Plot
            if plot:
                plt.plot(xlm.time_since_beginning, xlm.Dmb)
                xlm_na = xlm.dropna(subset=['time_since_beginning', 'forceActivated'])
                plt.plot(xlm_na.time_since_beginning[xlm_na.forceActivated], xlm_na.Df[xlm_na.forceActivated], linewidth=3)


def export_tif(config_segment, exports_list, select="",
               force_overwrite=False, verbose=True):
    """Wrapper function to export TIFF movies.
    
    Parameters:
    - select (str, default: ''): a string pattern used to only process some files. 
      For instance 'Pos0DC0'.    
    """
    
    ## unwrap variables
    img_p = config_segment['img_p']
    trk_p = config_segment['trk_p']
    csv_p = config_segment['csv_p']
    prefix = config_segment['prefix']
    directory = config_segment['directory']
    
    def print_(x, verbose, **kwargs):
        if verbose:
            print(x, **kwargs)
    
    def _export_tif(im, im1, da, ip, cp,
                    center=None, rotated=None, save1d=False, cropsize=None,
                    force_overwrite=False, verbose=True):
        """A sub-wrapper function"""
        params={'center':center, 'rotated':rotated,'save1d':save1d,'cropsize':cropsize}
        print_("  Exporting: {}".format(",\t".join(["{}:{}".format(*i) for i in params.items()])), verbose)
        suf = 'c{}'.format(center[0].upper()+center[1:])
        if not save1d:
            if rotated:
                op = os.path.join(prefix, directory, img_p[i]).replace(".ome.", "_rot_{}.".format(suf))
            else:
                op = os.path.join(prefix, directory, img_p[i]).replace(".ome.", "_norot_{}.".format(suf))
            if os.path.isfile(op) and not force_overwrite:
                return
            r = generate_movie(im, da, full=True, rotate=rotated,
                               cropsize=cropsize, center=center)
            with TiffWriter(op, imagej=True) as tif:
                tif.save(r[1].astype(np.float32))
        else:
            if rotated:
                op = os.path.join(prefix, directory, img_p[i]).replace(".ome.", "_rot1d_{}.".format(suf))
            else:
                op = os.path.join(prefix, directory, img_p[i]).replace(".ome.", "_norot1d_{}.".format(suf))
            if os.path.isfile(op) and not force_overwrite:
                return
            r = generate_movie(im, da, ch=[0,1,2,3], full=False, rotate=rotated,
                               cropsize=cropsize, center=center)
            with TiffWriter(op, imagej=True) as tif:
                tif.save(r[1].astype(np.float32))
        print_("  Saved file: {}".format(os.path.basename(op)), verbose)

    ## ==== Main function
    for i in range(len(img_p)):
        if select not in img_p[i]:
            continue
        print_("processing {}/{}: {}".format(i+1, len(img_p), img_p[i]), verbose)
        
        im, im1, da, tr, ex, ip, cp, ap = read_data(prefix, directory, i, img_p, trk_p, csv_p, verbose=False) ## Load data
        cp2 = cp.replace('.csv', '_3.csv')
        if not os.path.isfile(cp2):
            warn("Skipping {} - File not found".format(os.path.basename(cp2)))
            continue
        da = pd.read_csv(cp2)

        for j in exports_list:
            _export_tif(im, im1, da, ip, cp, **j,
                        force_overwrite=force_overwrite, verbose=verbose)
    print_("Done!", verbose)
        
def export_init(config):
    """Initializes the creation of an export folder."""
    f_dir = os.path.join(config['prefix'], config['directory'])
    e_dir = os.path.join(config['prefix'], config['directory'].replace('drift-correction/', 'exports/'))
    
    ## Create folder architecture
    if not os.path.isdir(e_dir):
        os.makedirs(e_dir)
        
    ## List files to copy & copy them
    tif_l = [i for i in os.listdir(f_dir) if i.endswith('_MAX.ome.tif')]
    csv3_l = []
    excl = []
    for t in tif_l:
        c = t.replace('_MAX.ome.tif', '_3.csv').replace('DC', '_cell')
        if not os.path.isfile(os.path.join(f_dir, c)):
            warn("TIF file {} does not have a _3.csv file".format(c))
            excl.append(c)
        else:
            csv3_l.append(c)
            shutil.copyfile(os.path.join(f_dir, c), os.path.join(e_dir, c.replace('_3.csv', '.csv')))
    
    # here we can do more things
    return e_dir, excl

        
def export_overlay(img_p, exports_list, excl, config, normalize=True, force_overwrite=False, verbose=True):
    """
    This wrapper function:
    Loops over all selected files and add add an overlay and/or normalized version.
    """
    def print_(x, verbose, **kwargs):
        if verbose:
            print(x, **kwargs)
    excl_tif = [i.replace("_cell", "DC").replace("_3.csv", ".ome.tif") for i in excl]
            
    for i in range(len(img_p)):
        print_("processing {}/{}: {}".format(i+1, len(img_p), img_p[i]), verbose)
        if img_p[i] in excl_tif:
            warn("File {} found in `excl`, skipped".format(img_p[i]))
            continue
    
        for j in exports_list:
            center = j['center']
            suf = 'c{}'.format(center[0].upper()+center[1:])
            if j['rotated']:
                suf = '_rot{}'+suf
            else:
                suf = '_norot{}'+suf
            if j['save1d']:
                suf=suf.format('1d_')
            else:
                suf=suf.format('_')
            suf+='.tif'
            print('  '+suf)
            
            fn = img_p[i].replace('.ome.tif', suf)
            fn_csv = img_p[i].replace('.ome.tif', '_3.csv').replace('DC', '_cell')
            ip = os.path.join(config['prefix'], config['directory'], fn)
            op = os.path.join(config['directory_export'], fn)
            cp = os.path.join(config['prefix'], config['directory'], fn_csv)
            assert os.path.isfile(ip), "File {} not found".format(ip)
            assert os.path.isfile(cp), "File {} not found".format(cp)
            
            if force_overwrite or not os.path.isfile(op): # Here check if the file exists (in exports/)
                if normalize:
                    norm='norm'
                    opp = op.replace('.tif', '_norm.tif')
                else:
                    norm=''
                    opp=op
                subprocess.call(['python3', './fiji_scripts/add_overlay.py', ip, op, cp, norm])
                assert os.path.isfile(opp), 'File {} was not created.'.format(opp)
        

def finetracking_init(config):
    """Initializes the creation of an export folder."""
    f_dir = os.path.join(config['prefix'], config['directory'])
    e_dir = os.path.join(config['prefix'], config['directory'].replace('drift-correction/', 'fine-correction/'))
    
    ## Create folder architecture
    if not os.path.isdir(e_dir):
        os.makedirs(e_dir)

## ==== Helper functions
def warn(msg):
    """An utterly simple wrapper to display a warning"""
    warnings.warn_explicit(msg, category=UserWarning, filename="here", lineno=0)

def has_screen():
    """An experimental function to quickly test whether ssh -X is active. 
    Not sure this is really working"""
    try:
        os.environ['DISPLAY']
        return True
    except:
        return False

def get_git_revision_short_hash():
    return str(subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])).replace("b'", "").replace("\\n'", "")

def get_git_revision_date():
    try:
        return str(subprocess.check_output("git log -1 --format=%cd origin/main".split())).replace("b'", "").replace("\\n'", "")
    except:
        return str(subprocess.check_output("git log -1 --format=%cd origin/master".split())).replace("b'", "").replace("\\n'", "")

def convert_path(p, system): 
    """We need this to deal with Windows paths :(
    From: https://stackoverflow.com/a/50924863/9734607b"""
    if "windows" in system:
        return "\\\\?\\{}".format(p.replace('/', '\\'))
    else:
        return p

## ==== CONFIG FILE SECTION ====
def has_dc_instructions(p):
    """Function that checks if a TIF file has drift-correction instructions attached to it."""
    if not os.path.isfile(p+'.txt'):
        return False
    else:
        return True

def load_config_file(p):
    """Loads a configuration file"""
    class multidict(OrderedDict):
        """Rename duplicated sections so that they can be imported separately
        From: https://stackoverflow.com/a/9888814"""
        _unique = 0   # class variable

        def __setitem__(self, key, val):
            if isinstance(val, dict):
                self._unique += 1
                key += str(self._unique)
            OrderedDict.__setitem__(self, key, val)
            
    parser = configparser.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    assert os.path.isfile(p+'.txt'), "File {} not found".format(p+'.txt')
    parser.read(p+'.txt')
    return parser

def get_number_cells(p):
    """Simply count the number of sections in the config .txt file. 
    `p` should not have the .txt extension"""
    parser = load_config_file(p)
    return len(parser.sections())


    
## ==== MAIN PREPROCESSING functions
def list_files_positions(prefix_path, result_path, use_analysis, lfn, version,
                         forceOn, verbose=True, save_log=True):
    """Creates an object suitable to loop over all the positions over several sequences"""
    
    ## === We first list the positions available for each sequence
    lfnP = {i:i.split('/')[-1] if len(i.split('/')[-1])>0 else i.split('/')[-2] for i in lfn}
    listPos = {}
    allPos = {}
    for j in lfn:
        lTif = [i for i in os.listdir(os.path.join(prefix_path, j)) if i.endswith(".ome.tif")]
        pos = get_positions(lTif)
        listPos[j]=pos
        allPos[j] = {k:v for (k,v) in zip(pos, lTif)}
    
    m = min([min(i) for i in listPos.values()])
    M = max([max(i) for i in listPos.values()])
    df_l = []
    for (k,v) in listPos.items():
        d = {i: i in v for i in range(M+1)}
        d["Sequence"]=k
        d["SequenceS"]=lfnP[k]
        df_l.append(d)
    pat = "_Pos{}.ome.tif"
    df = pd.DataFrame(df_l)
    df.set_index("Sequence", inplace=True)
    df_P = df.copy() # Just for display purpose
    df_P.replace([True, False], ["*", ""], inplace=True)
    
    # === Swap positions/folders
    posidict = {i:[os.path.join(j,allPos[j][i]) for j in list(df.index[df.loc[:,i]==True])] for i in range(m,M+1)}
    posidict = {k:v for (k,v) in posidict.items() if len(v)>0}
    
    # === Print stuff
    print("Available positions/sequence")
    print(df_P)

    # === Generate output folder
    if use_analysis != 'new':
        fn = os.path.join(pathlib.Path(os.path.join(result_path, os.path.dirname(lfn[0]))).parent, '{}'.format(use_analysis))
        assert os.path.isdir(fn), "Folder {} not found".format(fn)
    else:
        use_analysis = datetime.datetime.now().strftime("%Y%m%d")
        fn = os.path.join(pathlib.Path(os.path.join(result_path, os.path.dirname(lfn[0]))).parent, '{}'.format(use_analysis))
        i=0
        while os.path.isdir(fn):
            if i==0:
                fn = fn+'.{}'.format(i)
            fn = fn[:-2]+'.{}'.format(i)
            i+=1
        os.makedirs(fn)

    ## ==== Print state
    print("\nAvailable analyses: "+", ".join(os.listdir(os.path.join(pathlib.Path(os.path.join(result_path, os.path.dirname(lfn[0]))).parent))))
    print("Using folder: {}".format(use_analysis))
    print("Found {} files across {} positions".format(sum([len(i) for i in allPos.values()]), M+1))

    if save_log: ## Log (versions, etc)
        with open(os.path.join(fn, "pipeline_version.txt"), 'a') as f: ## Export pipeline version
            f.write("{}: Working with the version {} (commit {}), last updated on {}".format(datetime.datetime.now(), version, str(get_git_revision_short_hash()), get_git_revision_date()))

        with open(os.path.join(fn, "concatenation.txt"), 'a') as f: ## Export concatenation instruction
            f.write("files:\n")
            for i in lfn:
                f.write(i+"\n")
            f.write(str(lfn))
            f.write("\nTimestamps:\n")
            f.write(str(forceOn))

    if verbose:
        print(fn)
    return posidict, fn

def get_positions(l, final=True):
    """Extract the position number from a list of filenames
    If `final=True`, looks for the pattern: "PosX.ome.tif. 
    Else, looks for the pattern: "_PosX_". """
    fn = [os.path.basename(i) for i in l]
    patternF="\_Pos(\d+)\.ome\.tif$" # Extract position number using regexp
    patternNF="\_pos(\d+)\_" # Extract position number using regexp // Notice the lowercase 'pos'
    r = []
    if final=='any':
        for i in fn:
            tr = None
            try:
                tr = int(re.findall(patternNF, i)[0])
            except:
                try:
                    tr = int(re.findall(patternF, i)[0])
                except:
                    pass
            r.append(tr)
    elif final:
        try:
            r = [int(re.findall(patternF, i)[0]) for i in fn]
        except:
            print(fn)
            print(patternF)
    else:
        try:
            r = [int(re.findall(patternNF, i)[0]) for i in fn]
        except:
            print(fn)
            print(patternF)

    return r

## === TIMESTAMPS EXTRACTION

def save_timestamps(prefix_path, fn, posidict, system, fieldOn=None, force_overwrite=True):
    """Extracting the timestamps"""
    print("Saving to: {}".format(fn))
    ddf = {}
    for (k,v) in posidict.items():
        of = os.path.join(fn, 'timestamps_Pos{}.xls'.format(k))
        #if os.path.isfile(of) and not force_overwrite:
        #    continue
        print("Processing position {}/{} with {} files...".format(k, max(posidict.keys()), len(v)))
        clear_output(wait=True)
        df = get_timestamps_table(prefix_path, v, t_field=fieldOn, t_fieldOff=None, system=system) # Extract timestamps
        ddf[k]=df.copy()
        if not os.path.isfile(of) or force_overwrite:
            df.to_excel(of)
    print("Done!")
    return ddf

def get_timestamps_table(base_path, lfn, t_field, t_fieldOff, system):
    """t_field: list of lists (t_start, t_stop) in datetime format, without errors
    t_fieldOff: None, deprecated"""

    # sanity checks and transition checks
    assert type(t_field) not in (float, int), "t_field should be a list of lists"
    assert t_fieldOff is None, "t_fieldOff is deprecated, please set it to None"
    
    ## Get all start times
    start_times = []
    for p in lfn:
        try:
            tif = TiffFile(convert_path(os.path.join(base_path, p), system))
        except Exception as e:
            print(p)
            raise e
        mt = tif.micromanager_metadata
        s = mt['summary']['StartTime']
        nt = mt['summary']['IntendedDimensions']['time'] # number of frames (number of time points)
        if nt==1:
            tp = [tif[0].micromanager_metadata['ElapsedTime-ms']/1000.]
        else:
            frame_idx = []
            for i in range(nt):
                try:
                    frame_idx.append(mt['index_map']['frame'].index(i))
                except:
                    break

            tp = [tif[i].micromanager_metadata['ElapsedTime-ms']/1000. for i in frame_idx] # Save time points in seconds

        start_times.append({'path': p, 'start_time': s, 'timepoints': tp})

    ## Print times properly
    da = []
    for d in start_times:
        for t in d['timepoints']:
            d['time_in_file']=t
            da.append(d.copy())
    ## Assemble the table
    t_fieldOn = ",".join([i[0].strftime("%Y-%m-%d %H:%M:%S") for i in t_field])
    t_fieldOff = ",".join([i[1].strftime("%Y-%m-%d %H:%M:%S") for i in t_field])
    df=pd.DataFrame(da)
    df['start_time_s']=[time.mktime(time.strptime(i,'%Y-%m-%d %H:%M:%S %z')) for i in df.start_time.values]    
    df['time']=df.start_time_s+df.time_in_file
    df['timestamp'] = [datetime.datetime.fromtimestamp(i) for i in df.time.values]
    t_start = [datetime.datetime.fromtimestamp(i) for i in df.time.values]
    #df['timeOn']=df.time-t_fieldOn
    #df['timeOff']=df.time-t_fieldOff
    df['timeOn']=t_fieldOn
    df['timeOff']=t_fieldOff
    df['forceActivated']=[any([i[0]<=t and i[1]>=t for i in t_field]) for t in t_start]
    #df['forceActivated']=(df.timeOn>0)&(df.timeOff<0)
    df['time_since_beginning']=df.time-df.time[0]
    df['time_interval']=df.time.diff()
    df['seconds_since_first_magnet_ON']=[(i-t_field[0][0]).total_seconds() for i in t_start]
    df=df.sort_values(by='time')
    df['frame']=range(1, df.shape[0]+1)
    del df['timepoints']
    
    df['positions']=get_positions(df.path.values) # Extract position number
    
    return df


## ==== DBG functions
def manual_concatenation():
    """ /!\ Emergency concat"""
    emergency = False
    if emergency:
        for np in [6,10,11,15,17,18,19]: # Positions to concatenate
            p = ["/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191126-before/concatenated_Pos{}.ome.tif",
                 "/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191114_U2OS_stTetR-mCherry_GFP-ferritin_attraction_4bis/20191114_U2OS_stTetR-mCherry_GFP-ferritin_attraction_4_MMStack_Pos{}.ome.tif",
                 "/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191126-after/concatenated_Pos{}.ome.tif"]
            out_file = "/home/umr3664/CoulonLab/CoulonLab Dropbox/data/Maxime/concatenation/Veer/20191114 - fixed/20191126-final/concatenated_Pos{}.ome.tif".format(np)
            r = subprocess.call([config['fiji_path'], "concat.py"]+[os.path.join(config['prefix_path'], i.format(np)) for i in p]+[out_file, tf])


## ==== FORCE FIELD ====
def load_forcefield(prefix_path, ff_p, which='pillar1', z=22):
    """Load the force field at a given z plane
    which: one of ('pillar1', 'pillar2', 'simulated')
    prefix_path: the prefix path,
    z: the z plane to extract (for experimental force fields only)"""
    ff_ptmp = ff_p
    assert which in ff_ptmp.keys(), "'which' should be a string, among: {}".format(tuple(ff_ptmp.keys()))
    print("Reading {}".format(os.path.basename(ff_ptmp[which][0])))
    
    ff_p = {}
    for (k,p) in ff_ptmp.items():
        ff_p[k] = {'rect': os.path.join(prefix_path, p[0]+'.txt'), 'xyz': os.path.join(prefix_path,p[0]),
                   'ndims': p[1], 'has_cfg': p[2], 'simulated_pillar_pointing_down': p[3]}

    if ff_p[which]['ndims']==4: ## Load field
        im = TiffFile(ff_p[which]['xyz']).asarray()[z,:,:,:]
    elif ff_p[which]['ndims']==3:
        im = TiffFile(ff_p[which]['xyz']).asarray()[z,:,:]
    elif ff_p[which]['ndims']==2:
        im = TiffFile(ff_p[which]['xyz']).asarray()[:,:]
    else:
        raise TypeError("The loaded matrix has the wrong number of dimensions: {}".format(im.shape))
        
    if ff_p[which]['has_cfg']: ## Load rectangle
        cfg = configparser.ConfigParser() # Load the rectangle from the config file
        assert os.path.isfile(ff_p[which]['rect']), "Config file {} not found".format(ff_p[which]['rect'])
        cfg.read(ff_p[which]['rect'])
        r_force = ast.literal_eval(cfg[cfg.sections()[0]]['rect'])
        px_size = cfg[cfg.sections()[0]]['pixXY']
        simulated_pillar_pointing_down = ff_p[which]['simulated_pillar_pointing_down']
    else: # Deprecated, should not be used.
        ss = (im.shape[-2]-1, im.shape[-1]-1)
        r_force = ((0,ss[0]),(ss[0]/2,ss[1]/2), ss)

    return im, r_force, px_size, simulated_pillar_pointing_down

def rescale_field(src, src_px, dst_px, src_rect, verbose=True):
    """Rescales an image (a force field) to match a pixel size
    src: (matrix) a 2D/3D matrix
    src_px: the pixel size of the force field
    dst_px: the pixel size of the acquired images
    src_rect: the localization of the pillar (a polygon described as a list of list). 
        Its coordinates will also be rescaled"""
    scl = float(src_px)/dst_px
    new_size = list(src.shape[:-2])+[int(i*scl) for i in src.shape[-2:]] # we plan to rescale the XY coordinates only
    dst = skimage.transform.resize(src, new_size)
    dst_rect = [(int(i[0]*scl), int(i[1]*scl)) for i in src_rect]
    
    if verbose:
        print("Field px size: {}, Experiment px size: {}, zooming the force field by a factor: {}".format(src_px, dst_px, scl))
    return dst, dst_rect

def get_registered_pillar(r_force, r_chromag, img, pillar_location='top', out_shape='default', flip_force_field=False, debug=False):
    """A new version of `get_registered_pillar`
    A function to register the pillar onto the image. The algorithm is explained in
    more details in the `Force field registration.ipynb` notebook. 
    
    Briefly:
    1. Registration is performed using user-specified rectangles (`r_force` 
       and `r_chromag`, one tip of the rectangle matches the visible tip of 
       the pillar. The other  dimensions of the rectangle are not relevant. 
       The right tip of the rectangle is specified using the `pillar_location` 
       variable.
    2. Thus, we want to translate the reference force field to match the
       location and orientation of the tip of the pillar. 
    3. In practice, we first determine the rigid (rotation+translation) transform
       that maps ont tip of the rectangle onto the other. Then, we apply 
       this transform to the pillar image; `img`. To do this, we also need to
       know the size of the output image that needs to be produced. 
       - If `out_shape=='default', then the resulting image will have shape `img.shape*scale`
       - If `out_shape==(xshape,yshape), then this shape will be the resulting 
         shape. In that case, make sure that the final shape is bigger than 
         the initial one, else, the behaviour will be unexpected."""
    
    ## Extract triangles
    if flip_force_field: # should be done when the simulated pillar is pointing down
        r_force = r_force[::-1] 
    assert len(r_force) == 3, "The force should already be a triplet of points, not a rectangle"
    if pillar_location=='top':
        r = r_chromag[1:]
    elif pillar_location=='bottom':
        r = [r_chromag[-1]]+list(r_chromag)[:-2]
    else:
        raise NotImplemented("`pillar_location` should be either 'top' or 'bottom'")
    
    rf = np.array(r_force)
    rc = np.array(r)

    ## Normalize triangles    
    def norm_triplet(r, scl=80):
        """Normalize a triplet of points so that they can be overlaid perfectly
        """
        res = r.copy()
        res[:2,:] = scl*((res[:2,:]-res[1,:])/np.linalg.norm(np.diff(res[:2,:], axis=0))) + r[1,:]
        res[1:,:] = scl*((res[1:,:]-res[1,:])/np.linalg.norm(np.diff(res[1:,:], axis=0))) + r[1,:]        
        return res
    
    rfS = norm_triplet(rf)
    rcS = norm_triplet(rc)

    ## Estimate transform
    T = skimage.transform.EuclideanTransform()
    T.estimate(rcS, rfS)
    
    ## warp image
    if out_shape=='default':
        o_s = None
    else:
        o_s = out_shape
    I = skimage.transform.warp(img, T, output_shape=o_s)    
    
    if debug:
        plt.subplot(121)
        plt.plot(rc[:,0], rc[:,1])
        plt.plot(rcS[:,0], rcS[:,1]) 
        plt.scatter(rc[0,0], rc[0,1])        
        plt.plot(rf[:,0], rf[:,1])
        plt.plot(rfS[:,0], rfS[:,1])
        plt.scatter(rf[0,0], rf[0,1])
        plt.imshow(img)
        plt.title("Before transform")
        
        plt.subplot(122)
        plt.plot(rc[:,0], rc[:,1])
        plt.plot(rcS[:,0], rcS[:,1])         
        plt.scatter(rc[0,0], rc[0,1])
        plt.title("After transform")
        plt.imshow(I)

    if I.max()==0: # Sanity check: non-zero force field
        print("ERRROR REGGISSTTTREEERING!!!")
        raise IOError("CASSSE TOUT")
    return I,T

def get_Fstart(p, pDC, c, imgF, px_size, r_force, config):
    """
    Extract the initial force applied to the locus.
    
    Returns: Fx, Fy, Fz, Fm
    - with Fm the norm of the (Fx, Fy, Fz) vector
    """

    ## Fstart_per_MNP_fN
    cfg_raw = load_config_file(p)
    cfg = parse_config_file(cfg_raw, c)
    assert (cfg['x0'] is not None) and (cfg['y0'] is not None)
    x = np.array([cfg['x0']])
    y = np.array([cfg['y0']])
    
    r = map_force(imgF, x, y, px_size, cfg['pixXY'], r_force, cfg['pillar'], config['simulated_pillar_pointing_down'])
    F_per_MNP = (r[1][0]**2+r[1][1]**2+r[1][2]**2)**0.5

    ## MNPs_per_locus
    # Get the variables we need
    p_ts = p.replace("concatenated", "timestamps").replace(".ome.tif", ".xls")
    ts = pd.read_excel(p_ts)    
    ts_s = ts.seconds_since_first_magnet_ON
    t=ts.frame[ts_s <= 0].max()-1 # 0-indexed
    x0 = cfg['x0']-cfg['xLoc']+cfg['winCrop']/2.
    y0 = cfg['y0']-cfg['yLoc']+cfg['winCrop']/2.
    p_img = re.sub('\.ome.tif$', 'DC{}.ome.tif'.format(c), pDC)
    
    # Get reference
    cal = config['mnp_calibration']
    pos_ref,c_ref = re.findall('_Pos(\d+)DC(\d)\.', cal['movie'])[-1]
    p_ref = re.sub('_Pos\d+\.', '_Pos{}DC{}.'.format(pos_ref, c_ref), pDC)
    x_ref = cal['x']-1 #cfg_ref['x0']-cfg_ref['xLoc']+cfg_ref['winCrop']/2.
    y_ref = cal['y']-1 #cfg_ref['y0']-cfg_ref['yLoc']+cfg_ref['winCrop']/2.

    # Perform the fits (reference + current file)
    f_ref = track_single(p_ref, cal['frame']-1, x_ref, y_ref, verbose=False)
    f_img = track_single(p_img, t, x0, y0) # we might need to take `chLocus` into account

    # Calibrate fit
    I_ref = f_ref['Fluo. instensity'].values[0]
    I_img = f_img['Fluo. instensity'].values[0]    
    nMNP = I_img/I_ref*cal['MNPs']
    F_start_per_locus_pN = F_per_MNP[0]*nMNP/1000.

    # fit the locus intensity + background
    # get ADU value -> maybe we should get it from the datasets.cfg file
    return r[1], F_per_MNP, nMNP, F_start_per_locus_pN

def track_single(path_tif, frame, x0, y0, z0=14, chLocus=1, verbose=False):
    """Detect with subpixel accuracy a single point
    This is an ugly wrapper honestly..."""
    
    # write a .mtrk file
    mtrk=path_tif.replace(".ome.tif", ".cal.mtrk")
    with open(mtrk, 'w') as f:
        f.write("# T X Y Z\n")
        f.write("{} {} {} {}".format(frame+1, x0+1, y0+1, z0+1))

    ## Perform fit
    trk.trackUsingMouse(path_tif, '_', mtrk_file=mtrk, psfPx=3.0, psfZPx=1., offsets=None, thresholdSD=5,
                        force2D=False, extractZ=False, channel2D=chLocus, fnTif3D=path_tif, fps=-1, border=3, disp=2, verbose=verbose, skipwarnings=not verbose)
    
    ## Retrieve results
    trk2=mtrk.replace(".mtrk", ".trk2")
    assert os.path.isfile(trk2)
    res=pd.read_csv(trk2, sep='\t')
    l=res[res['# Frame']==frame]
    
    #assert l["Code (bits: spot detected, fit convereged)"].values[0]==3.0, "Fit did not converge, file: {}\n initial point: f={}, x={}, y={}, z={}".format(path_tif, frame+1, x0+1, y0+1, z0+1)
    if l["Code (bits: spot detected, fit convereged)"].values[0]!=3.0:
        print("Fit did not converge, file: {}\n initial point: f={}, x={}, y={}, z={}".format(path_tif, frame+1, x0+1, y0+1, z0+1))
    return l

## ==== HANDLE ALL FILES
def load_cell(fn, p, pDC, c=0, track_type='trackmate',
              src_only=False, load_bp=False, verbose=True, apply_shifts=True,
              trdr_only=False, mtrk_cvfn=None):
    """Here:
        1. we load the configuration file, 
        2. the full movie, 
        3. check that the DC movies were computed. Else, we issue a warning.
        4. make sure that we have tracking information
        5. load the tracks
    p:              the path of the configuration (.txt) file
    pDC:            the path of the drift-corrected image
    track_type:     where the tracking comes from, either `trackmate` or `trackusingmouse` or None (no tracking)
    verbose (bool): should we talk a lot?  
    load_bp (bool): should we load the bandpassed image, or just the source file?
    src_only (bool):should we only load the source movie and the .txt file?
    apply_drifts(bool): should we correct the pixel coordinates by the ones in the .shifts file?
    """

    ptxt=p

    if load_bp:
        p=p.replace(".ome.", "BP.ome.")
        
    ## Check inputs
    track_types = ('trackmate', 'trackusingmouse', None)
    assert track_type in track_types, "`track_type` should be in one of {}".format(track_types)
    if mtrk_cvfn is not None:
        assert os.path.isfile(mtrk_cvfn), "mtrk_cvfn: file {} not found".format(mtrk_cvfn)

    ## Load configuration
    cfg_raw = load_config_file(ptxt) # Configuration file
    cfg_n = len(cfg_raw.sections())
    if cfg_n<c:
        raise TypeError("c={} but file {} has only {} sections".format(c, p+'.txt', cfg_n))
    cfg = parse_config_file(cfg_raw, c) # returns a dict
    xLoc = cfg['xLoc']
    yLoc = cfg['yLoc']
    wc = cfg['wc']
    
    ## Load tracking data // /!\ Note MW: these dictionary contain one element, remove them
    dc_p = {}
    tr_p = {}
    trF_p = {} # used for track using mouse only
    tr_o = {} # used to store offsets

    #s=c ## TODO MW /!\ Temporary debug by MW
    pdc = pDC.replace('.ome.', 'DC'+str(c)+'.ome.')
    if track_type is None: # Not loading a tracking file
        ptr = None
        ptrF = None
        offsets = None
    elif track_type == 'trackmate':
        ptr = pDC.replace('.ome.tif', 'DC'+str(c)+'_Tracks.xml')
        offsets = None
    elif track_type == 'trackusingmouse':
        mtrk, offsets, code = load_mtrk(fn=fn, img_p=pdc, mtrk_cvfn=mtrk_cvfn,
                                       fix_roi=True, cfg_p=ptxt, cfg_c=c)
        if code[0] != 'ok':
            return {'flag': code[1]}
        ptr = mtrk
        ptrF = mtrk.replace(".mtrk", ".trk2")

    if track_type is None:
        print("WARNING: No tracking type has been specified, no tracking loaded")
    else:
        if not os.path.isfile(pdc):
            print("WARNING: File {} does not exist".format(pdc))
        if not os.path.isfile(ptr):
            print("WARNING: File {} does not exist".format(ptr))
            
    dc_p[c] = pdc
    tr_p[c] = ptr
    trF_p[c] = ptrF
    tr_o[c] = offsets

    if not trdr_only:
        img = TiffFile(p).asarray() ## Load the movie
    else:
        img = None
    if src_only:
        r1 =  {'img': img,
               'rect2': [xLoc-wc, yLoc-wc, xLoc+wc, yLoc+wc],
               'rect': [[xLoc-wc, xLoc-wc, xLoc+wc, xLoc+wc, xLoc-wc],
                        [yLoc-wc, yLoc+wc, yLoc+wc, yLoc-wc, yLoc-wc]],
               'flag': 'ok'}
        r1.update(cfg)
        return r1

    if not os.path.isfile(tr_p[c]): # check if the mtrk file is elsewhere
        print("File does not exist")
        return {'flag': 'fileerror'}

    ## Load the tracking data
    if track_type=='trackmate':
        tr = load_trackmate(tr_p[c], um_to_px=1/cfg['pixXY'], um_to_pxZ=1/cfg['stepZ']) ## Load the tracking data
    elif track_type == 'trackusingmouse':
        tr = pd.read_csv(tr_p[c], sep=' ', names=['t','x', 'y', 'z'], skiprows=1)
        tr.drop_duplicates(subset='t', keep='last', inplace=True)
        tr['t']-=1
        tr['x_orig'] = tr['x']#*pixXY
        tr['y_orig'] = tr['y']#*pixXY
        tr['z_orig'] = tr['z']#*stepZ
        if os.path.isfile(trF_p[c]):
            print("fitted", trF_p[c])
            del tr['x']
            del tr['y']
            del tr['z']
            trF = pd.read_csv(trF_p[c], sep='\t')
            trF.rename(columns={'# Frame': 't', 'Position X': 'x', 'Position Y': 'y', 'Position Z': 'z'}, inplace=True)
            tr = pd.merge(tr, trF, on='t')
        else:
            print("Could not find fitting `.trk2` file. WARNING!")
    if apply_shifts:
        dr = pd.read_csv(dc_p[c]+'.shifts') # Loads a simple drift file of the form frame,x,y,z
        tr_dr = pd.merge(tr, dr, left_on='t', right_on='frame', suffixes=('', '_drift'))
    elif tr_o[c] is not None and not os.path.isfile(trF_p[c]): # apply offsets, so far cannot be combined with shifts, .trk2 files are already offset corrected in our setting...
        print(tr_p[c])
        print("applying offsets")
        if apply_shifts:
            print("BIG BIG WARNING")
        print(tr_o[c])
        tr_dr = pd.merge(tr, tr_o[c], left_on='t', right_on='frame', suffixes=('', '_drift'))
        tr_dr.x -= tr_dr.x_drift
        tr_dr.y -= tr_dr.y_drift
        tr_dr.z -= tr_dr.z_drift
    else: # We apply a shift (drift correction) of 0 px
        tr_dr = tr.copy()
        tr_dr['frame']=tr_dr['t']
        tr_dr['x_drift']=0
        tr_dr['y_drift']=0
        tr_dr['z_drift']=0
    tr_dr['x_DC']=tr_dr['x']-tr_dr['x_drift']+xLoc-wc
    tr_dr['y_DC']=tr_dr['y']-tr_dr['y_drift']+yLoc-wc
    tr_dr['z_DC']=tr_dr['z']-tr_dr['z_drift']
    tr_dr['x_DCum']=tr_dr['x_DC']*cfg['pixXY']
    tr_dr['y_DCum']=tr_dr['y_DC']*cfg['pixXY']
    tr_dr['z_DCum']=tr_dr['z_DC']*cfg['stepZ']
    tr_dr['pixelsize']=cfg['pixXY']
    tr_dr['chDNA'] = cfg['chDNA']
    tr_dr['zstep']=cfg['stepZ']
    tr_dr["fitOk"]=tr_dr["Code (bits: spot detected, fit convereged)"]==3.0
    
    if verbose:
        print("Sequence {} has ({}/{}) drift-computed regions and ({}/{}) tracked loci".format(os.path.basename(p), len(dc_p), cfg_n, len(tr_p), cfg_n))
        print("Pixel size in XY is {}µm, and pixel size in Z is {}µm.".format(cfg['pixXY'], cfg['stepZ']))
        print("DNA channel: {}".format(cfg['chDNA']))

    r2 = {'img': img, 'tr_dr': tr_dr,
          'rect2': [xLoc-wc, yLoc-wc, xLoc+wc, yLoc+wc],
          'rect': [[xLoc-wc, xLoc-wc, xLoc+wc, xLoc+wc, xLoc-wc],
                   [yLoc-wc, yLoc+wc, yLoc+wc, yLoc-wc, yLoc-wc]],
          'flag': 'ok'}
    r2.update(cfg)
    return r2

def load_mtrk(fn, img_p, mtrk_cvfn=None, fix_roi=False,
              cfg_p=None, cfg_c=None, verbose=False):
    """It is getting increasingly complex to properly load a .mtrk file, provided
    the multiple sources that exist, and the corrections to apply.

    Variable:
      img_p (str): the path to the image
      mtrk_csvfn (str): the path to the mtrk reattribution table, if the .mtrk file
                        is not located next to the img_p file.
      fix_roi (bool): if set to yes, the script will load the .txt cfg file to offset
                      the ROI coordinates. Use this if the tracking was performed on the
                      full (concatenated frame) instead of on the cropped frame.
      cfg_p (str): the path to the .txt file that contains ROI and pixel size info
      cfg_c (int): the cell index to load in the cfg file

    Returns: (mtrk_path, offsets)
      mtrk_path (str): the path to an existing .mtrk file
      offsets (Pandas dataframe): an offset table, that allows the transformation of the 
                                  .mtrk coordinates to usable coordinates
    """
    ## Create variables
    mtrk_fn_default = img_p.replace('.ome.tif', '.ome_.mtrk')

    ## Sanity checks
    if not os.path.isfile(mtrk_fn_default) and mtrk_cvfn is None:
        #raise IOError(".mtrk file not found, {} do not exist and `mtrk_cvfn` not provided.".format(mtrk_fn_default, mtrk_cvfn))
        return None, None, ('error', ".mtrk file not found, {} do not exist and `mtrk_cvfn` not provided.".format(mtrk_fn_default, mtrk_cvfn))
    elif not os.path.isfile(mtrk_fn_default) and not os.path.isfile(mtrk_cvfn):
        #raise IOError(".mtrk file not found, {} and {} do not exist.".format(mtrk_fn_default, mtrk_cvfn))
        return None, None, ("error", ".mtrk file not found, {} and {} do not exist.".format(mtrk_fn_default, mtrk_cvfn))
    if fix_roi and ((cfg_p is None) or (type(cfg_c) is not int)):
        #raise TypeError("`fix_roi` set to `True` but `cfg_p` and/or `cfg_c` not provided")
        return None, None, ("error", "`fix_roi` set to `True` but `cfg_p` and/or `cfg_c` not provided")
    elif fix_roi and not os.path.isfile(cfg_p+'.txt'):
        #raise IOError("`cfg_p` {} does not exists".format(cfg_p+'.txt'))
        return None, None, "error", "`cfg_p` {} does not exists".format(cfg_p+'.txt')
    elif verbose and not os.path.isfile(cfg_p+'.txt'):
        print("Not loading a `cfg_p` file")

    ## Create the mtrk_fn variable (locate an existing .mtrk file)
    mtrk_fn = mtrk_fn_default

    if os.path.isfile(mtrk_fn_default): #os.path.isfile(mtrk_cvfn), not totally the same:
        if verbose:
            print("Loading mtrk file: {}".format(mtrk_fn))
            print("`fix_roi` set to False")
        fix_roi=False # we do not fix the roi in this specific setup kindof a hack
        
    else:
        if verbose:
            print("Loading conversion file: {}".format(mtrk_cvfn))
        tb = pd.read_csv(mtrk_cvfn)
        o_mtrk = os.path.basename(mtrk_fn_default)
        tbd = {i.strip():(j.strip(),int(k)) for i,j,k in zip(tb.cropped_frame.values, tb.full_frame.values, tb.cell)}
        n_mtrk = tbd[o_mtrk][0]
        c = tbd[o_mtrk][1]
        mtrk_fn = os.path.join(fn, n_mtrk)

    ## Load .mtrk file
    mtrk_tb = pd.read_csv(mtrk_fn, sep=" ", names=('t','x', 'y','z'), skiprows=1) # the .mtrk, uncorrected

    ## Generate offsets variable
    nframes = mtrk_tb.t.max()
    offsets = pd.DataFrame({'frame' : range(nframes), 
                            'x': np.zeros(nframes), 
                            'y': np.zeros(nframes), 
                            'z': np.zeros(nframes)})

    ## TODO MW: replace with parse_config_file
    if fix_roi: # read the config file & extract xLoc, yLoc 
        cfg = load_config_file(cfg_p)
        ns = cfg.sections()[cfg_c] # name section
        xLoc = cfg.getint(ns, 'xLoc')
        yLoc = cfg.getint(ns, 'yLoc')
        winCrop = cfg.getint(ns, 'winCrop')

        # Set offsets variable
        offsets.x += xLoc-int(winCrop/2)
        offsets.y += yLoc-int(winCrop/2)        

    return mtrk_fn, offsets, ('ok', ':)') 

def parse_config_file(cfg, c):
    """Loads a .txt file associated with a cell"""
    sec = cfg.sections()[c]
    winCrop = cfg.getint(sec, 'winCrop')
    x0,y0 = (None,None)
    if cfg.has_option(sec, 'x0') and cfg.has_option(sec, 'y0'):
        x0 = cfg.getint(sec, 'x0')-1
        y0 = cfg.getint(sec, 'y0')-1
        
    r = {'sec' : sec,
         'pillar' : ast.literal_eval(cfg.get(sec, 'rect')),
         'chDNA' : cfg.getint(sec, 'chDNA')-1,
         'zLoc' : cfg.getint(sec, 'zLoc')-1,
         'xLoc' : cfg.getint(sec, 'xLoc')-1,
         'yLoc' : cfg.getint(sec, 'yLoc')-1,
         'winCrop' : winCrop,
         'pixXY' : cfg.getfloat(sec, 'pixXY'),
         'stepZ' : cfg.getfloat(sec, 'stepZ'),
         'wc' : int(winCrop/2.),
         'x0' : x0,
         'y0' : y0
    }
    return r

def read_data(prefix_path, directory, i, img_p, trk_p, csv_p, verbose=True):
    """Read the following stuff:
    - the CSV file
    - the contour
    - the location of the pillar"""
    cp = os.path.join(prefix_path, directory, csv_p[i])
    tp = os.path.join(prefix_path, directory, trk_p[i])
    ip = os.path.join(prefix_path, directory, img_p[i])
    ep = ip.replace(".ome.tif", "_excl.txt")
    ap = ip.replace('.ome.tif', '_quant')
    
    if verbose:
        print("Reading {}".format(os.path.basename(ip)))
    da = pd.read_csv(cp)
    if os.path.isfile(ep):
        with open(ep, 'r') as f:
            ex = [i[:-1] for i in f.readlines()]
    else:
        ex = []
    #tr = pd.read_csv(tp, names=['t','x','y','z'], header=None, sep=" ", skiprows=1)
    #tr.drop_duplicates(inplace=True, keep='last', subset='t')
    #tr.t-=1
    tr = da.loc[:,['t','x','y','z', 'x_orig']]
    tr=tr[tr.x_orig>10] # filter track
    del tr['x_orig']
    
    with TiffFile(ip.replace('.ome', '_seg.ome')) as tif:
        im = tif.asarray()
    with TiffFile(ip.replace('.ome', '_seg1d.ome')) as tif:
        im1 = tif.asarray()        
    return (im, im1, da, tr, ex, ip, cp, ap)


## ==== DATA ANALYSIS
def compute_metrics(prefix_path, directory, index, img_p, trk_p, csv_p,
                    tgt_window_size=5, scl=500,
                    select_f=[], verbose_ch=1, verbose=False, verbose_sbs=True # DEBUG OPTIONS
                   ):
    """A function that compute many metrics for a given cell/movie
    Parameters:
        - tgt_window_size (int): the size (in pixels) of the window to compute the
          tangent to the nucleus
        - scl (float): a plotting parameter to rescale the force vector
        - select_f (list of ints, default: [-1]): a list of ints used to process only some frames in a file.
    """
    
    wsi = tgt_window_size
    im, im1, da, tr, ex, ip, cp, ap = read_data(prefix_path, directory, index, img_p, trk_p, csv_p) # Load some data for the rest of the analysis
    TZ = {int(k):int(v) for (k,v) in zip(tr.t.values, tr.z.values)}
    if len(ex)>0 and ex[0]=='all': # we skip the entire movie
        return {"err": "Movie skipped based on `_excl.txt` file"}
    else:
        ex = [int(i) for i in ex if i not in ['', '\n']]
    tb_size = (im.shape[0]//5+1, 5)
    if verbose:
        plt.figure(figsize=(18,18))

    Dmb = []
    x00, y00 = da.x.values[0], da.y.values[0]

    j=1
    for f in TZ.keys(): #
        if (select_f != []) and (-1 not in select_f) and (f not in select_f):
            print("Frame skipped based on `select_f`")
            continue
        if f in ex:
            print("-- Frame {} planned to be skipped, based on `_excl.txt` file".format(f))
            #continue
        z = TZ[f]
        t = da[(da.x>10)&(da.y>10)&(da.t==f)]
        if t.shape[0]==0: # No tracking data for this time point
            print("Dataframe for this time ({}) point has {} entries".format(f, t.shape[0]))
            continue

        x0, y0 = t.x.values[0], t.y.values[0]
        Fx = t.Fx.values[0]
        Fy = t.Fy.values[0]
        Fx_m = t.Fx_m.values[0]
        Fy_m = t.Fy_m.values[0]
        
        angle = angle_between(np.array([Fx_m, Fy_m]), np.array([0,-1]))/np.pi*180
        if Fx==0 and Fy==0:
            print("Force is zero")
            x1,y1,x2,y2=np.nan, np.nan, np.nan, np.nan
            model=(np.nan,np.nan)
        else:
            mask=im[f,z,3,:,:]
            if mask.max()==0:
                print("Frame {}: no segmented cell".format(f))
                continue
            elif mask[int(y0),int(x0)]==0:
                print("Frame {}: the initial point ({},{}) is not in the mask/cell".format(f, int(x0), int(y0)))
                continue
            try:
                ci = mask.max() # /!\ fix this...
                x1,y1, err1 = find_intersection(mask, x0, y0, Fx, Fy, cell_index=ci, towards_magnet=True)
                x2,y2, err2 = find_intersection(mask, x0, y0, Fx, Fy, cell_index=ci, towards_magnet=False)
                x3,y3, err3 = find_intersection(mask, x0, y0, -Fy, Fx, cell_index=ci, towards_magnet=True)
                x4,y4, err4 = find_intersection(mask, x0, y0, -Fy, Fx, cell_index=ci, towards_magnet=False)
            except Exception as e:
                print("Frame {}".format(f))
                raise e
            wrn = [err1, err2, err3, err4]
            if verbose and not all([i==[] for i in wrn]): # process warnings
                tmp = [(i,"no error") if j==[] else (i,j) for (i,j) in enumerate(wrn)]
                for (kk,mm) in tmp:
                    if mm !="no error":
                        print("Frame {}: warning {}: {}".format(f,kk+1,mm))
                
            msk = masks_to_outlines(im1[f,3,:,:], usena=False)
            msk-=msk.min()
            ext = msk[max(0,(y1-wsi-1)):(y1+wsi),max(0,(x1-wsi-1)):(x1+wsi)] # extract a window around the intersection, of size wsi
            wh = np.argwhere(ext==ext.max()) # recover coordinates of the contour
            wh[:,0]+=y1-wsi
            wh[:,1]+=x1-wsi
            model_fit = np.polyfit(wh[:,0]+np.random.normal(0, 1e-4, size=wh.shape[0]), wh[:,1], 1) # fit straight line, add small eps to avoid poorly conditioned fit
            
        F = np.array([[Fx,Fy]])
        F /= np.linalg.norm(F)
        
        Dmb.append({'t':f, 
                    'Dmb': np.sqrt((x1-x0)**2+(y1-y0)**2), # Distance travelled
                    'Df' : np.dot(np.array([[x0-x00, y0-y00]]), F.T)[0][0], # displacement along force, /!\ wrt first frame
                    'x_mb': x1,
                    'y_mb': y1,
                    'x_Nmb': x2,
                    'y_Nmb': y2,
                    'x_omb': x3,
                    'y_omb': y3,
                    'x_oNmb': x4,
                    'y_oNmb': y4,
                    'a_tgt': model_fit[0],
                    'b_tgt': model_fit[1],
                    'angle': angle,
                   }) 

    if Dmb==[]:
        return {"err": "Empty dataframe produced (no frame was successively processed)"}
    Dmbs = pd.DataFrame(Dmb)
    trm = pd.merge(da, Dmbs, on='t', how='left')
    coords = extract_coordinates_system(trm, im1[:,3],plot=False)
    trmc = pd.merge(trm, coords, on='t', how='left')
    trmc.to_csv(cp.replace(".csv", "_2.csv"))

    if verbose:
        plot_metrics(im, trmc, verbose_ch, ip, sidebyside=verbose_sbs)
    
    return {"trmc": trmc, "im": im, "err": ""}

def generate_movie(im, trmc, ch=[1,2], full=False, rotate=True,
                   cropsize=None, center='centroid', use_angle_median=False):
    """Function that extracts a movie and centers it around the centroid 
    (ctrX, ctrY) columns. Rotation is optional (argument `rotate`)"""

    assert max(ch)<im.shape[1]
    cx,cy=im.shape[-2], im.shape[-1]
    if full:
        out = np.zeros((im.shape[0], im.shape[1], im.shape[2]+1, 2*cx,2*cy))
        ch_pts = out.shape[2]-1
    else:
        out = np.zeros((im.shape[0], len(ch)+1, 2*cx,2*cy))
        ch_pts = len(ch)
    TZ = {int(k):int(v) for (k,v) in zip(trmc.t.values, trmc.z.values)}
    TC = {int(k):(int(v), int(w)) for (k,v,w) in zip(trmc.t.values, trmc.ctrX.values, trmc.ctrY.values) if (not np.isnan(v) and not np.isnan(w))}
    TP = {int(k):(int(v), int(w)) for (k,v,w) in zip(trmc.t.values, trmc.x.values, trmc.y.values) if (not np.isnan(v) and not np.isnan(w))}
    if center=='centroid':
        v1=100
        v2=200
    else:
        v2=100
        v1=200
        tmp=TP
        TP=TC
        TC=tmp
    for f in TC.keys():
        if full:
            out[f, :, ch_pts, (cx-1):(cx+1), (cy-1):(cy+1)]=v1 # centroid/locus is now centered
            if f in TP:
                out[f, :, ch_pts, cx-TC[f][1]+TP[f][1], cy-TC[f][0]+TP[f][0]]+=v2 # position of the locus/centroid
            out[f, :, 0:ch_pts, (cx-TC[f][1]):(2*cx-TC[f][1]), (cy-TC[f][0]):(2*cy-TC[f][0])]=im[f]
        else:
            out[f, ch_pts, (cx-1):(cx+1), (cy-1):(cy+1)]=v1 # centroid/locus is now centered
            if f in TP:
                out[f, ch_pts, cx-TC[f][1]+TP[f][1], cy-TC[f][0]+TP[f][0]]+=v2 # position of the locus/centroid
            for (ci, c) in enumerate(ch):
                out[f, ci, (cx-TC[f][1]):(2*cx-TC[f][1]), (cy-TC[f][0]):(2*cy-TC[f][0])]=im[f,TZ[f], c]

    r = np.zeros_like(out)
    if full:
        sli3=[slice(0,r.shape[i]) for i in range(2)]+\
            [slice(0,r.shape[2]-2)]+\
            [slice(0,r.shape[i]) for i in range(3,len(r.shape))]
        sli0=[slice(0,r.shape[i]) for i in range(2)]+\
            [slice(r.shape[2]-2,r.shape[2])]+\
            [slice(0,r.shape[i]) for i in range(3,len(r.shape))]
    else:
        sli3=[slice(0,r.shape[i]) for i in range(1)]+\
            [slice(0,r.shape[1]-2)]+\
            [slice(0,r.shape[i]) for i in range(2,len(r.shape))]
        sli0=[slice(0,r.shape[i]) for i in range(1)]+\
            [slice(r.shape[1]-2,r.shape[1])]+\
            [slice(0,r.shape[i]) for i in range(2,len(r.shape))]

    ## Rotation
    if rotate:
        if use_angle_median:
            angle = -trmc.angle.mean()
        else:
            angle = -trmc.angle0s.mean()
        scipy.ndimage.rotate(out[sli0], angle, output=r[sli0],
                             reshape=False, axes=(len(out.shape)-2, len(out.shape)-1),
                             order=0)
        scipy.ndimage.rotate(out[sli3], angle, output=r[sli3],
                             reshape=False, axes=(len(out.shape)-2, len(out.shape)-1),
                             order=3)
    else:
        r=out
        
    ## Crop
    if cropsize is None:
        return [r, r]
    else:
        Cx=r.shape[-2]/2
        Cy=r.shape[-1]/2
        sli=tuple([slice(0,r.shape[i]) for i in range(len(r.shape)-2)]+[slice(int(Cx-cropsize[0]/2), int(Cx+cropsize[0]/2)), slice(int(Cy-cropsize[1]/2), int(Cy+cropsize[1]/2))])
        return [r, r[sli]]

def find_intersection(mask, x0, y0, Fx, Fy, towards_magnet=True, cell_index=1):
    """Find the intersection between the contour of the cell and a line going
    towards/away from the magnet, starting at the locus (x0,y0) location, 
    in the direction of the force (Fx,Fy)
    """
    err = []
    x,y = int(round(x0)), int(round(y0))
    cell_index=mask[y,x]
    f = lambda x: y0+(x-x0)*Fy/Fx
    
    nc = 0
    while mask[y,x]==cell_index:
        ox,oy=x,y
        if (Fx>0 and towards_magnet) or (Fx<0 and not towards_magnet):
            x+=1
        elif (Fx<0 and towards_magnet) or (Fx>0 and not towards_magnet):
            x-=1
        else:
            print("Error, Fx=0")
        y = int(round(f(x)))
        if not (x>=0 and x<mask.shape[1] and y>=0 and y<mask.shape[0]): # should we clip the data?
            #print("Frame {}: The new point is out of the frame... ({}, {})".format(x,y))
            err.append("The new point is out of the frame... ({}, {})".format(x,y))
            x = min(max(0,x),mask.shape[1]-1)
            y = min(max(0,y),mask.shape[0]-1)
            break
    if abs(x-ox)<=1 and abs(y-oy)>=3:
        while mask[oy,ox]==cell_index:
            if oy<y:
                oy+=1
            elif oy>y:
                oy-=1
            if mask.shape[1]<=(oy+1) or y==oy :
                break
    return(ox,oy,err)

def extract_coordinates_system(trm, masks, basis2d=False, ctr=None, extract_clos=True, plot=False):
    """Function that extracts a (3D, orthonormal) basis based on the direction of the force,
    and an origin based on the centroid of the segmentation.
    If basis2d==True, the z component of the force is ignored."""

    ## Sanity checks
    if masks is not None:
        assert masks.shape[0]==trm.shape[0] # we have the right number of frames
    else:
        assert trm.shape[0]==len(ctr)
    bs2d=[1,0][basis2d]
        
        
    ## Precomputations
    xy = {i.t: (int(round(i.x)),int(round(i.y))) for (_,i) in trm.iterrows()}
    
    ## Extract the basis
    Fx=trm.Fx.copy()
    Fy=trm.Fy.copy()
    Fz=trm.Fz.copy()*bs2d

    n1 = np.sqrt(Fx**2+Fy**2+Fz**2)
    B1x = Fx/n1
    B1y = Fy/n1
    B1z = Fz/n1

    n2 = np.sqrt(B1x**2+B1y**2) # Orthogonal to B1, in the (x,y) plane
    B2x = -B1y/n2
    B2y = B1x/n2
    B2z = B1x*0

    B3x = -B1z*B1x # Vector product B1^B2
    B3y = -B1z*B1y-B1x*B1x
    B3z = B1x*B1x+B1y*B1y
    
    basis = pd.DataFrame({'t': trm.t,
                         'B1x': B1x, 'B1y': B1y, 'B1z': B1z, 
                         'B2x': B2x, 'B2y': B2y, 'B2z': B2z, 
                         'B3x': B3x, 'B3y': B3y, 'B3z': B3z})    
    
    if plot:
        ax = plt.subplot(111)
        x,y=xy[0]
        scl=25
        ax.plot((0,Fx[0]*scl),(0, Fy[0]*scl), linewidth=3)
        ax.plot((x,x+B1x[0]*scl),(y, y+B1y[0]*scl), label='B1')
        ax.plot((x,x+B2x[0]*scl),(y, y+B2y[0]*scl), label='B2')
        ax.set_aspect('equal')

    ## Extract basis over time
    Fx_m = trm.Fx_m.values[0]
    Fy_m = trm.Fy_m.values[0]
    Fz_m = trm.Fz_m.values[0]*bs2d
    
    n1m = np.sqrt(Fx_m**2+Fy_m**2+Fz_m**2)
    B1xm = Fx_m/n1m
    B1ym = Fy_m/n1m
    B1zm = Fz_m/n1m

    n2m = np.sqrt(B1xm**2+B1ym**2) # Orthogonal to B1, in the (x,y) plane
    B2xm = -B1ym/n2m
    B2ym = B1xm/n2m
    B2zm = B1xm*0

    B3xm = -B1zm*B1xm # Vector product B1^B2
    B3ym = -B1zm*B1ym-B1xm*B1xm
    B3zm = B1xm*B1xm+B1ym*B1ym
    
    Pm = np.matrix([[B1xm, B2xm],
                   [B1ym, B2ym]]).T # matrice de passage
    basis_m = pd.DataFrame([{'B1xm': B1xm, 'B1ym': B1ym, 'B1zm': B1zm,
                             'B2xm': B2xm, 'B2ym': B2ym, 'B2zm': B2zm,
                             'B3xm': B3xm, 'B3ym': B3ym, 'B3zm': B3zm}]*basis.shape[0])

    B_1 = np.array([B1x, B1y]).T
    B_2 = np.array([B2x, B2y]).T
    assert np.isnan(B1xm*B2xm+B1ym*B2ym) or (B1xm*B2xm+B1ym*B2ym<1e-10), "B1m and B2m are not orthogonal: {}".format(B1xm*B2xm+B1ym*B2ym)
    for i in range(B1x.shape[0]):
        assert np.isnan(B_1[i,0]*B_2[i,0]+B_1[i,1]*B_2[i,1]) or (B_1[i,0]*B_2[i,0]+B_1[i,1]*B_2[i,1]<1e-10), "B1 and B2 are not orthogonal: {}".format(B_1[i,0]*B_2[i,0]+B_1[i,1]*B_2[i,1])
        
    ## Extract the centroid
    if ctr is None:
        ctr = []
        for i in range(masks.shape[0]):
            x,y=xy[i]
            cell_index = masks[i,y,x]
            if cell_index==0: # the initial point is not inside the cell
                ## TODO MW: here include the excluded frames
                print(f"frame {i}: the locus ({x}, {y}) is not inside the cell")
                if plot:
                    plt.imshow(masks[i])
                    plt.scatter((x),(y))
                    plt.scatter((y),(x))
                    plt.show()
                continue
            coord = np.argwhere(masks[i]==cell_index)
            coord_m = coord.mean(axis=0)
            ctr.append({'t':i, 'ctrX': coord_m[1], 'ctrY': coord_m[0]})
    else:
        ctr = [{'t': i, 'ctrX': ctr[i][0], 'ctrY': ctr[i][1]} for i in range(len(ctr))]
        
    centroid = pd.DataFrame(ctr)

    if plot:
        plt.scatter(coord[:,0], coord[:,1], s=5, alpha=0.1, label='cell contour')
        plt.scatter(coord_m[0], coord_m[1], color='red', label='centroid')
        
    ## Extract the point closest to the magnet
    clos = []
    for i in range(trm.shape[0]):
        if not extract_clos:
            clos.append({'t': i, 'closX': np.nan, 'closY': np.nan})
        else:
            x,y=xy[i]
            cell_index = masks[i,y,x]
            if cell_index==0: # the initial point is not inside the cell
                ## TODO MW: here include the excluded frames
                continue
            coord = np.argwhere(masks[i]==cell_index)
            b = basis.loc[basis.t==i]
            p = trm.loc[trm.t==i]
            assert b.shape[0]==1 and p.shape[0]==1

            P = np.matrix([[b.B1x.values[0], b.B2x.values[0]], [b.B1y.values[0], b.B2y.values[0]]]).T # matrice de passage
            coord2 = np.asarray(P.dot(coord.T).T) # Coordinates in the new reference frame
            #coord2[:,0]-=p.x.values[0] # origin of the locus as the reference frame
            #coord2[:,1]-=p.y.values[0]
            idx = np.argwhere(coord2[:,1]==coord2[:,1].max())[0]
            #closD = coord2[idx, 1][0] # Distance between the locus and the closest point to the magnet
            closP = coord[idx,:] # coordinates of the point in the main reference frame
            clos.append({'t': i, 'closX': closP[0,0], 'closY': closP[0,1]})
    closest = pd.DataFrame(clos)
    
    if plot:
        x,y=xy[0]
        plt.scatter(x,y, color='grey', s=20, label='locus')
        plt.scatter(closP[0,0], closP[0,1], color='black', s=5, label='closest point to magnet')
        plt.legend()

    m = pd.merge(pd.merge(basis, centroid, how='left', on='t'), closest, how='left', on='t')
    m = pd.concat([m, basis_m], axis=1)
    
    # Location with respect to the centroid
    m['x_ctr'] = trm.x-m.ctrX
    m['y_ctr'] = trm.y-m.ctrY
    m['x_clos']= trm.x-m.closX
    m['y_clos']= trm.y-m.closY
    m['closD'] = np.sqrt((trm.x-m.closX)**2+(trm.y-m.closY)**2)

    # Coordinates in the fixed coordinate system (u,v)
    #coord = np.array([m.x_ctr, m.y_ctr]) # 2 rows x n cols
    #coord2 = np.asarray(Pm.dot(coord.T).T)
    #m['u_ctr'] = coord2[:,0]
    #m['v_ctr'] = coord2[:,1]

    # Coordinates in the rotating coordinate system (uu,vv)
    uu_ctr = []
    vv_ctr = []
    u_ctr = [] # Should be vectorized
    v_ctr = []
    for (i,l) in m.iterrows():
        coord = np.array([l.x_ctr, l.y_ctr])
        P = np.matrix([[l.B1x, l.B2x], [l.B1y, l.B2y]]).T # matrice de passage
        coord2 = np.asarray(P.dot(coord.T).T)
        coord2m = np.asarray(Pm.dot(coord.T).T)
        uu_ctr.append(coord2[0][0])
        vv_ctr.append(coord2[1][0])
        u_ctr.append (coord2m[0][0])
        v_ctr.append (coord2m[1][0])
    m['uu_ctr'] = uu_ctr
    m['vv_ctr'] = vv_ctr
    m['u_ctr'] = u_ctr
    m['v_ctr'] = v_ctr
    
    return m

def angle_between(v1, v2):
    """Oriented angle betweent two vectors
    From: https://stackoverflow.com/a/61533633/9734607"""
    a=np.array(v1)
    b=np.array(v2)
    cosTh = np.dot(a,b)
    sinTh = np.cross(a,b)
    #return np.rad2deg(np.arctan2(sinTh,cosTh))
    return np.arctan2(sinTh,cosTh) # returns in radians



## === PLOTS
def masks_to_outlines(masks, usena=False):
    """Copied and adapted from cellpose software"""
    Ly, Lx = masks.shape
    nmask = masks.max()
    outlines = np.zeros((Ly,Lx), np.bool)
    # pad T0 and mask by 2
    T = np.zeros((Ly+4)*(Lx+4), np.int32)
    Lx += 4
    iun = np.unique(masks)[1:]
    for iu in iun:
        y,x = np.nonzero(masks==iu)
        y+=1
        x+=1
        T[y*Lx + x] = 1
        T[y*Lx + x] =  (T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                        T[y*Lx + x-1]     + T[y*Lx + x+1] )
        outlines[y-1,x-1] = np.logical_and(T[y*Lx+x]>0 , T[y*Lx+x]<4)
        T[y*Lx + x] = 0
    #outlines *= masks
    outlines = outlines.astype(float)+1
    if usena:
        outlines[outlines==0]=np.nan
    return outlines
