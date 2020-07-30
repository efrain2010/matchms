from typing import Generator
from typing import List
import numpy
from ..Spectrum import Spectrum


def parse_msp_file(filename: str) -> List[dict]:
    """Read msp file and parse info in list of spectrum dictionaries."""

    # List that will contain all the differente "spectrums"
    spectrums = []

    # Lists/dicts that will contain all params, masses and intensities of each molecule
    params = {}
    masses = []
    intensities = []

    # Peaks counter. Used to track and count the number of peaks
    peakscount = 0

    with open(filename, 'r') as f:
        for line in f:
            rline = line.rstrip()

            if len(rline) == 0:
                continue

            if ':' in rline:
                # Obtaining the params
                splitted_line = rline.split(":", 1)
                if splitted_line[0].lower() == 'comments':
                    # Obtaining the parameters inside the comments index
                    for s in splitted_line[1][2:-1].split('" "'):
                        splitted_line = s.split("=", 1)
                        if splitted_line[0].lower() in params.keys() and splitted_line[0].lower() == 'smiles':
                            params[splitted_line[0].lower()+"_2"] = splitted_line[1].strip()
                        else:
                            params[splitted_line[0].lower()] = splitted_line[1].strip()
                else:
                    params[splitted_line[0].lower()] = splitted_line[1].strip()
            else:
                # Obtaining the masses and intensities
                peakscount += 1

                splitted_line = rline.split(" ")

                masses.append(float(splitted_line[0]))
                intensities.append(float(splitted_line[1]))

                # Obtaining the masses and intensities
                if int(params['num peaks']) == peakscount:
                    peakscount = 0
                    spectrums.append(
                        {
                            'params': params,
                            'm/z array': numpy.array(masses, dtype="float"),
                            'intensity array': numpy.array(intensities, dtype="float")
                        }
                    )
                    params = {}
                    masses = []
                    intensities = []

    return spectrums


def load_from_msp(filename: str) -> Generator[Spectrum, None, None]:
    """
    MSP file to a Spectrum object
    Function that reads a .msp file and converts the info
    in Spectrum objects.
    Parameters
    ----------
    filename : str
        path of the msp file
    Returns
    -------
    Spectrum
        Yield a spectrum object with the data of the msp file

    Example:
    .. code-block:: python

    from matchms.importing import load_from_msp

    spectrum = load_from_msp("MoNA-export-GC-MS-first10.msp")
    """

    for pyteomics_spectrum in parse_msp_file(filename):
        metadata = pyteomics_spectrum.get("params", None)
        mz = pyteomics_spectrum["m/z array"]
        intensities = pyteomics_spectrum["intensity array"]

        # Sort by mz (if not sorted already)
        if not numpy.all(mz[:-1] <= mz[1:]):
            idx_sorted = numpy.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz, intensities=intensities, metadata=metadata)
