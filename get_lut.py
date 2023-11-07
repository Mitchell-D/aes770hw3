""" """
#from sbdart_info import default_params
from pathlib import Path
import numpy as np
import pickle as pkl
import zarr
#from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

from krttdkit.acquire.sbdart import dispatch_sbdart, parse_iout
import krttdkit.visualize.guitools as gt
import krttdkit.visualize.geoplot as gp

def flux_over_fields(fields, coords, sbdart_args, tmp_dir_parent,
                     workers=5):
    assert len(fields)==len(coords)
    dims = tuple(len(f) for f in fields)
    # Get a list of all argument combinations in argument space, ie all
    # combinations of (i, j, k, l, ...) indeces of each 'coords' list, in order
    arg_space = np.full([len(coords[i]) for i in range(len(fields))], True)
    coord_idxs = list(zip(*np.where(arg_space)))
    # Make a SBDART argument dictionary with each combination
    fields_coords = [
            {fields[i]:coords[i][aidx[i]] for i in range(len(aidx))}
            for aidx in coord_idxs
            ]
    no_repeat_rint = np.random.default_rng().choice(
            int(1e6), size=len(fields_coords), replace=False)
    tmp_dirs = [tmp_dir_parent.joinpath(f"tmp_{hash(str(abs(x)))}")
                for x in no_repeat_rint]
    # Add default args. At least as of Python 3.10.9, the rightmost
    # elements in a dict expansion get priority.
    full_args = (({**sbdart_args, "iout":10, **fields_coords[i]},
                  tmp_dirs[i])
            for i in range(len(fields_coords)))

    ## Make an array with the same dimensionality as provided coords for each
    ## of the returned parameters. This is a bit wasteful since wlinf and wlsup
    ## are probably the same in any case.
    flux_labels = ['wlinf', 'wlsup', 'ffew', 'topdn', 'topup',
                   'topdir', 'botdn', 'botup', 'botdir']
    flux_arrays = [np.full([len(c) for c in coords],np.nan,dtype=float)
                   for i in range(len(flux_labels))]
    try:
        with Pool(workers) as pool:
            for sb_in,sb_out in pool.imap(_mp_sbdart, full_args):
                ## Get the nd array index corresponding to this run
                idx = tuple(coords[fields.index(f)].index(sb_in[0][f])
                            for f in fields)
                ## Add this run's entry to each output array
                for i in range(len(sb_out["flux"])):
                    flux_arrays[i][*idx] = sb_out["flux"][i]
    except Exception as E:
        raise E
    return flux_labels, flux_arrays

#({'idatm': 2, 'isalb': 7, 'wlinf': 0.4, 'wlsup': 5, 'wlinc': 0.05, 'iaer': 5, 'wlbaer': 0.64, 'wbaer': 0.945, 'gbaer': 0.58, 'iout': 10, 'sza': 0, 'tbaer': 0}, PosixPath('buffer/sbdart/tmp_-6172520305277131742'))
#{'flux': (0.4, 5.0, 4.6, 1262.5, 54.585, 1262.5, 1001.7, 25.066, 960.73), 'flux_labels': ('wlinf', 'wlsup', 'ffew', 'topdn', 'topup', 'topdir', 'botdn', 'botup', 'botdir')}

def _mp_sbdart(args):
    """
    Run SBDART given a temporary directory, and a dict of SBDART style
    arguments that defines an output type (key "iout").

    Return the output from krttdkit.acquire.sbdart.parse_iout given the args.

    This method always removes its temporary directory, even if it errors out.

    :@param args: tuple like (sbdart_args:dict, tmp_dir:Path) identical to
        the parameters of krttdkit.acquire.sbdart.dispatch_sbdart. It is
        enforced that tmp_dir doesn't exist, and that a specific output style
        "iout" is provided.
    :@param return: 2-tuple like (args, output) where args is identical to the
        provided arguments (for asynchronous processing posterity), and output
        is from krttdkit.acquire.sbdart.parse_iout if the model was effectively
        executed. The format of the output depends on the iout specified in
        the sbdart_args dictionary.
    """
    sbdart_args, tmp_dir = args
    assert not tmp_dir.exists()
    assert tmp_dir.parent.is_dir()
    assert "iout" in sbdart_args.keys()
    try:
        return (args, parse_iout(
            sb_out=dispatch_sbdart(sbdart_args, tmp_dir),
            iout_id=sbdart_args["iout"], print_stdout=True))
    except Exception as E:
        raise E
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

if __name__=="__main__":
    tmp_dir = Path("buffer/sbdart")
    #zarr_file = Path("data/aerosol_lut.zip")
    #flux_file = Path("data/aerosol_sfluxes.pkl")
    #zarr_file = Path("buffer/tmp_lut.zip")
    flux_file = Path("buffer/tmp_flux.pkl")
    #description = "Spectral response of several boundary layer AODs in DESIS' wave range; no atmospheric scattering (since L2 data); fixed solar zenith of 23.719deg; rural aerosol types; default aerosol profile"
    description = "Multiple aerosol loadings over vegetation; normal atmospheric effects"

    """
    """
    sbdart_args = {
            "btemp":300,
            "idatm":2, # Mid-latitude summer
            #"idatm":1, # Tropical
            #"pbar":0, # no atmospheric scattering or absorption
            #"pbar":-1, # Default atmospheric scattering and absorption
            "isalb":7, # Ocean water

            "wlinf":.5,
            "wlsup":5,
            "wlinc":.05,

            #"wlinf":5,
            #"wlsup":50,
            #"wlinc":0.25,

            #"wlinc":0.1,
            "iaer":5, ## User-defined
            "wlbaer":.64,#",".join(map(str,[.42,.64,.86,.12])), ## single scatter albedo
            #"wbaer":.945,#",".join(map(str,[.95,.94,.93,.92])), ## single scatter albedo
            "gbaer":.58,#",".join(map(str,[.68,.59,.55,.54])), ## Assymetry parameter
            #"qbaer":.58,

            #"nre":9,
            #"zcloud":2,
            #"sza":20,
            }


    cods = list(np.logspace(np.log10(0.01), np.log10(40), 15))
    aods = [0, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2]
    nres = [4,5,6,7,8,9,10,11,12]
    szas = list(range(0, 70, 5))
    wbaers = [.8, .85, .9, .95, 1.]
    #fields=["sza", "tcloud"]
    #fields=["sza", "tbaer"]
    #fields=["nre", "tcloud"]
    fields=["wbaer", "tbaer"]
    coords=[
        wbaers,
        #szas,
        #nres,
        #cods,
        aods,
        ]

    '''
    fof = flux_over_fields(
            fields=fields,
            coords=coords,
            sbdart_args=sbdart_args,
            tmp_dir_parent=tmp_dir,
            workers=7,
            )
    pkl.dump(fof, flux_file.open("wb"))
    exit(0)
    '''

    fof = pkl.load(Path("buffer/flux_ssa-aod_sw.pkl").open("rb"))

    gp.plot_lines(
            coords[1],
            [fof[1][4][i,:] for i in range(len(coords[0]))],
            #labels=[f"SZA: {c}" for c in coords[0]], show=True,
            #labels=[f"CRE: {c}" for c in coords[0]], show=True,
            labels=[f"SSA: {c}" for c in coords[0]], show=True,
            plot_spec={
                #"title":"Model COD vs Outgoing SW (CRE=$9\,\mu m$; Over Ocean)",
                "title":"Model AOD vs Outgoing SW",
                #"xlabel":"Cloud Optical Depth",
                "xlabel":"Aerosol Optical Depth",
                "ylabel":"Outgoing Shortwave Flux ($W\,m^{-2}$)",
                #"ylabel":"Outgoing Longwave Flux ($W\,m^{-2}$)",
                })
