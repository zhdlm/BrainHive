from enum import Enum

class Microscope(Enum):
    ZEISS_INCUB = "Zeiss Incubator Microscope"
    ZEISS_FRAP = "Zeiss FRAP Microscope"
    ZEISS_CONFOCAL = "Zeiss Confocal Microscope"
    LEICA_IPSC = "Leica iPSC room"

class Magnification(Microscope):
    X10 = "x10"
    X5= "x5"