import gabase

GlobalArrays: {.pragma: gacommon, header: "ga.h".}

var C_CHAR* {.importc: "C_CHAR", gacommon.}: cint
var C_INT* {.importc: "C_INT", gacommon.}: cint
var C_FLOAT* {.importc: "C_FLOAT", gacommon.}: cint
var C_DBL* {.importc: "C_DBL", gacommon.}: cint

proc toGAType*(t: typedesc[char]): cint = C_CHAR
proc toGAType*(t: typedesc[int]): cint = C_INT
proc toGAType*(t: typedesc[float32]): cint = C_FLOAT
proc toGAType*(t: typedesc[float64]): cint = C_DBL

