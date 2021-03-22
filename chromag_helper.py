## ChroMag functions.
## By MW, Nov. 2019-March 2021
## GPLv3+

## General imports
import os, subprocess, datetime, re, pathlib, time
import pandas as pd
from IPython.display import clear_output
from skimage.external.tifffile import TiffFile

## ==== Helper functions
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
    return str(subprocess.check_output("git log -1 --format=%cd origin/main".split())).replace("b'", "").replace("\\n'", "")

def convert_path(p, system): 
    """We need this to deal with Windows paths :(
    From: https://stackoverflow.com/a/50924863/9734607b"""
    if "windows" in system:
        return "\\\\?\\{}".format(p.replace('/', '\\'))
    else:
        return p

## ==== MAIN PREPROCESSING functions
def list_files_positions(prefix_path, result_path, use_analysis, lfn):
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
    print("Found {} files across {} positions".format(sum([len(i) for i in allPos.values()]), M))

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

def save_timestamps(prefix_path, fn, posidict, system, fieldOn=None):
    """Extracting the timestamps"""
    print("Saving to: {}".format(fn))
    ddf = {}
    for (k,v) in posidict.items():
        of = os.path.join(fn, 'timestamps_Pos{}.xls'.format(k))
        print("Processing position {}/{} with {} files...".format(k, max(posidict.keys()), len(v)))
        clear_output(wait=True)
        df = get_timestamps_table(prefix_path, v, t_field=fieldOn, t_fieldOff=None, system=system) # Extract timestamps
        ddf[k]=df.copy()
        if not os.path.isfile(of):
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
        tif = TiffFile(convert_path(os.path.join(base_path, p), system))
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
