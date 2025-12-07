import std/[cmdline, os]

# gets C argc (including program name)
proc cargc*: cint = cint(paramCount() + 1)

proc cargv*(argc: cint): cstringArray =
  # gets C argv (including program name at index 0)
  var argv = newSeq[string](argc)
  argv[0] = getAppFilename()  # argv[0] is the program name
  for idx in 1..<argv.len: 
    argv[idx] = paramStr(idx)
  return allocCStringArray(argv)