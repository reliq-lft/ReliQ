#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/gaugeio.nim
  Contact: reliq-lft@proton.me

  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  MIT License
  
  Copyright (c) 2025 reliq-lft
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN 
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
]#

## Gauge Field I/O
## ===============
##
## This module provides high-level I/O facilities for gauge field configurations
## using the ``GaugeField`` type from ``gauge/gaugefield``. It handles:
##
## - Reading LIME/SciDAC/ILDG gauge configurations into ``GaugeField`` objects
## - Writing ``GaugeField`` objects to LIME/SciDAC files
## - Automatic validation: checksum, plaquette, and link trace are computed
##   and printed upon reading
##
## Example usage:
## ```nim
## import parallel
## import lattice
## import gauge/gaugefield
## import io/gaugeio
##
## parallel:
##   let lat = newSimpleCubicLattice([8, 8, 8, 16])
##   let ctx = newGaugeFieldContext[Complex64](gkSU3, rkFundamental)
##   var u = lat.newGaugeField(ctx)
##
##   readGaugeField(u, "config.lime")
##   # Checksum, plaquette, and link trace are printed automatically
##
##   writeGaugeField(u, "output.lime")
## ```

import std/[strformat, strutils, xmlparser, xmltree, endians, math]

import tensorio
import lime

import ../lattice
import ../tensor/[tensor]
import ../gauge/gaugefield
import ../globalarrays/[gawrap]
import ../utils/[complex]

export gaugefield

# ===========================================================================
# Gauge Field Validation Display
# ===========================================================================

proc parseStoredValue*(userXml: string, tag: string): (bool, float64) =
  ## Try to extract a float value from a named XML tag in the userXml string.
  ## Returns (found, value)
  if userXml.len == 0:
    return (false, 0.0)
  try:
    let xml = parseXml(userXml)
    let node = xml.child(tag)
    if node != nil:
      let text = node.innerText.strip()
      if text.len > 0:
        return (true, parseFloat(text))
  except:
    discard
  return (false, 0.0)

proc printGaugeValidation*(reader: TensorFieldReader, 
                           plaq: float64, lt: float64) =
  ## Print gauge field validation info: file metadata, record metadata,
  ## checksum status, plaquette, and link trace in one unified box.
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║                    Gauge Field I/O Summary                     ║"
  echo "╠════════════════════════════════════════════════════════════════╣"
  echo fmt"║  Version:     {reader.fileInfo.version:<48} ║"
  echo fmt"║  Spacetime:   {reader.fileInfo.spacetime:<48} ║"
  var dimStr = ""
  for i, d in reader.fileInfo.dims:
    if i > 0: dimStr.add " x "
    dimStr.add $d
  echo fmt"║  Dimensions:  {dimStr:<48} ║"
  echo fmt"║  Volume fmt:  {reader.fileInfo.volfmt:<48} ║"
  echo "╠════════════════════════════════════════════════════════════════╣"
  echo fmt"║  Datatype:    {reader.recordInfo.datatype:<48} ║"
  echo fmt"║  Precision:   {reader.recordInfo.precision:<48} ║"
  echo fmt"║  Colors:      {reader.recordInfo.colors:<48} ║"
  echo fmt"║  Spins:       {reader.recordInfo.spins:<48} ║"
  echo fmt"║  Type size:   {reader.recordInfo.typesize:<48} ║"
  echo fmt"║  Data count:  {reader.recordInfo.datacount:<48} ║"
  echo "╠════════════════════════════════════════════════════════════════╣"
  if reader.hasChecksum:
    let storedStr = fmt"suma={reader.storedChecksum.suma:08x}  sumb={reader.storedChecksum.sumb:08x}"
    let computedStr = fmt"suma={reader.computedChecksum.suma:08x}  sumb={reader.computedChecksum.sumb:08x}"
    echo fmt"║  Stored checksum:   {storedStr:<42} ║"
    echo fmt"║  Computed checksum: {computedStr:<42} ║"
    let csMatch = reader.storedChecksum.suma == reader.computedChecksum.suma and
                  reader.storedChecksum.sumb == reader.computedChecksum.sumb
    let csStatus = if csMatch: "PASSED ✓" else: "FAILED ✗"
    echo fmt"║  Checksum:          {csStatus:<42} ║"
  else:
    let csStatus = "N/A (not in file)"
    echo fmt"║  Checksum:          {csStatus:<42} ║"
  echo "╠════════════════════════════════════════════════════════════════╣"
  # Plaquette: show computed and compare to stored if available
  let plaqStr = fmt"{plaq:.15g}"
  echo fmt"║  Plaquette:         {plaqStr:<42} ║"
  let (hasStoredPlaq, storedPlaq) = parseStoredValue(reader.userXml, "plaquette")
  if hasStoredPlaq:
    let storedPlaqStr = fmt"{storedPlaq:.15g}"
    echo fmt"║  Stored plaquette:  {storedPlaqStr:<42} ║"
    let plaqDiff = abs(plaq - storedPlaq)
    let plaqTol = 1.0e-10
    let plaqOk = plaqDiff < plaqTol
    let plaqStatus = if plaqOk: fmt"PASSED ✓  (Δ = {plaqDiff:.2e})" else: fmt"FAILED ✗  (Δ = {plaqDiff:.2e})"
    echo fmt"║  Plaquette match:   {plaqStatus:<42} ║"
  # Link trace: show computed and compare to stored if available
  let ltStr = fmt"{lt:.15g}"
  echo fmt"║  Link trace:        {ltStr:<42} ║"
  let (hasStoredLt, storedLt) = parseStoredValue(reader.userXml, "linkTrace")
  if hasStoredLt:
    let storedLtStr = fmt"{storedLt:.15g}"
    echo fmt"║  Stored link trace: {storedLtStr:<42} ║"
    let ltDiff = abs(lt - storedLt)
    let ltTol = 1.0e-10
    let ltOk = ltDiff < ltTol
    let ltStatus = if ltOk: fmt"PASSED ✓  (Δ = {ltDiff:.2e})" else: fmt"FAILED ✗  (Δ = {ltDiff:.2e})"
    echo fmt"║  Link trace match:  {ltStatus:<42} ║"
  echo "╚════════════════════════════════════════════════════════════════╝"

# ===========================================================================
# Raw data plaquette/linkTrace computation (rank 0, no stencil needed)
# ===========================================================================

proc readMatrix(data: seq[byte], offset: int, nc: int, swap: bool): seq[Complex64] =
  ## Read an nc x nc complex matrix from binary data at given byte offset
  let nElems = nc * nc
  result = newSeq[Complex64](nElems)
  for e in 0..<nElems:
    var re, im: float64
    copyMem(addr re, unsafeAddr data[offset + e * 16], 8)
    copyMem(addr im, unsafeAddr data[offset + e * 16 + 8], 8)
    if swap:
      var reBE, imBE: float64
      swapEndian64(addr reBE, addr re)
      swapEndian64(addr imBE, addr im)
      re = reBE
      im = imBE
    result[e] = complex64(re, im)

proc matMul(a, b: seq[Complex64], nc: int): seq[Complex64] =
  ## Multiply two nc x nc complex matrices
  result = newSeq[Complex64](nc * nc)
  for i in 0..<nc:
    for j in 0..<nc:
      var s = complex64(0.0, 0.0)
      for k in 0..<nc:
        s = s + a[i * nc + k] * b[k * nc + j]
      result[i * nc + j] = s

proc matAdjoint(a: seq[Complex64], nc: int): seq[Complex64] =
  ## Compute adjoint (conjugate transpose) of nc x nc matrix
  result = newSeq[Complex64](nc * nc)
  for i in 0..<nc:
    for j in 0..<nc:
      result[i * nc + j] = conjugate(a[j * nc + i])

proc matTrace(a: seq[Complex64], nc: int): Complex64 =
  ## Compute trace of nc x nc matrix
  result = complex64(0.0, 0.0)
  for i in 0..<nc:
    result = result + a[i * nc + i]

proc computePlaquetteFromBinary*(data: seq[byte], dims: seq[int], nc: int, swap: bool): float64 =
  ## Compute plaquette from raw binary gauge data (big-endian SciDAC format).
  ## Works entirely in memory on one rank — no stencil or MPI needed.
  let D = dims.len
  assert D == 4, "Only 4D lattices supported"
  var globalVol = 1
  for d in dims: globalVol *= d
  let nElems = nc * nc
  let bytesPerDir = nElems * 16 # Complex64 = 16 bytes
  let bytesPerSite = D * bytesPerDir
  
  var traceSum = 0.0
  var nPlaq = 0
  
  # Compute strides for lexicographic index
  var strides: array[4, int]
  strides[0] = 1
  for i in 1..<D:
    strides[i] = strides[i-1] * dims[i-1]
  
  for site in 0..<globalVol:
    # Get coordinates of this site
    var coords: array[4, int]
    var idx = site
    for d in 0..<D:
      coords[d] = idx mod dims[d]
      idx = idx div dims[d]
    
    for mu in 1..<D:
      for nu in 0..<mu:
        # U_mu(x)
        let umu_x = readMatrix(data, site * bytesPerSite + mu * bytesPerDir, nc, swap)
        
        # U_nu(x+mu): forward neighbor in mu direction
        var fwdMuCoords = coords
        fwdMuCoords[mu] = (coords[mu] + 1) mod dims[mu]
        var fwdMuIdx = 0
        for d in 0..<D: fwdMuIdx += fwdMuCoords[d] * strides[d]
        let unu_xmu = readMatrix(data, fwdMuIdx * bytesPerSite + nu * bytesPerDir, nc, swap)
        
        # U_nu(x)
        let unu_x = readMatrix(data, site * bytesPerSite + nu * bytesPerDir, nc, swap)
        
        # U_mu(x+nu): forward neighbor in nu direction
        var fwdNuCoords = coords
        fwdNuCoords[nu] = (coords[nu] + 1) mod dims[nu]
        var fwdNuIdx = 0
        for d in 0..<D: fwdNuIdx += fwdNuCoords[d] * strides[d]
        let umu_xnu = readMatrix(data, fwdNuIdx * bytesPerSite + mu * bytesPerDir, nc, swap)
        
        # Plaquette = Tr[ U_mu(x) * U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x) ]
        let ta = matMul(umu_x, unu_xmu, nc)
        let tb = matMul(unu_x, umu_xnu, nc)
        let tbAdj = matAdjoint(tb, nc)
        let plaq = matMul(ta, tbAdj, nc)
        traceSum += matTrace(plaq, nc).re
        inc nPlaq
  
  result = traceSum / float64(nPlaq * nc)

proc computeLinkTraceFromBinary*(data: seq[byte], dims: seq[int], nc: int, swap: bool): float64 =
  ## Compute average link trace from raw binary gauge data.
  let D = dims.len
  var globalVol = 1
  for d in dims: globalVol *= d
  let nElems = nc * nc
  let bytesPerDir = nElems * 16
  let bytesPerSite = D * bytesPerDir
  
  var traceSum = 0.0
  for site in 0..<globalVol:
    for mu in 0..<D:
      let mat = readMatrix(data, site * bytesPerSite + mu * bytesPerDir, nc, swap)
      traceSum += matTrace(mat, nc).re
  
  result = traceSum / float64(D * nc * globalVol)

# ===========================================================================
# Read GaugeField from file
# ===========================================================================

proc readGaugeField*[D: static[int], L: Lattice[D], T](
  u: var GaugeField[D, L, T],
  filename: string,
  swapBytes: bool = true,
  validate: bool = true
) =
  ## Read a gauge configuration from a LIME/SciDAC/ILDG file into a GaugeField.
  ##
  ## On read, the checksum is verified, and the plaquette and link trace
  ## are computed and printed for validation.
  ##
  ## Parameters:
  ##   u: The GaugeField to read data into (must be pre-allocated)
  ##   filename: Path to the input file
  ##   swapBytes: Whether to swap bytes for endianness (default: true)
  ##   validate: Whether to print validation info (default: true)
  
  # Read the raw data into the underlying TensorField array
  # Suppress tensorio's own output — we print our own consolidated box
  readGaugeField(u.field, filename, swapBytes, printInfo = false)
  
  if validate:
    # Print unified validation info (rank 0 only)
    # Compute plaquette/linkTrace from raw binary data on rank 0
    # (distributed stencil does not yet support cross-rank halo exchange,
    #  so we compute directly from the file data on rank 0)
    if GA_Nodeid() == 0:
      let reader = newTensorFieldReader(filename)
      reader.loadBinaryData(validate = false)
      
      # Compute observables from the raw binary data
      let nc = reader.colors
      let dims = reader.dims
      let plaq = computePlaquetteFromBinary(reader.binaryData, dims, nc, swapBytes)
      let lt = computeLinkTraceFromBinary(reader.binaryData, dims, nc, swapBytes)
      
      printGaugeValidation(reader, plaq, lt)
      reader.close()

# ===========================================================================
# Write GaugeField to file
# ===========================================================================

proc makeGaugeUserXml*(plaq: float64, lt: float64, extra: string = ""): string =
  ## Build a user XML string containing plaquette and link trace values,
  ## plus any extra user-supplied content.
  var xml = "<?xml version=\"1.0\"?>\n<gaugeInfo>\n"
  xml.add fmt"  <plaquette>{plaq:.15g}</plaquette>" & "\n"
  xml.add fmt"  <linkTrace>{lt:.15g}</linkTrace>" & "\n"
  if extra.len > 0:
    xml.add "  " & extra & "\n"
  xml.add "</gaugeInfo>"
  return xml

proc writeGaugeField*[D: static[int], L: Lattice[D], T](
  u: GaugeField[D, L, T],
  filename: string,
  userXml: string = "",
  swapBytes: bool = true
): LimeStatus =
  ## Write a gauge configuration from a GaugeField to a LIME/SciDAC file.
  ##
  ## Plaquette and link trace are automatically computed from the written
  ## binary data and stored in the user XML record for validation on re-read.
  ## Any additional userXml content is appended.
  ##
  ## Parameters:
  ##   u: The GaugeField to write
  ##   filename: Path to the output file
  ##   userXml: Optional extra user metadata XML content
  ##   swapBytes: Whether to swap bytes for endianness (default: true)
  ##
  ## Returns:
  ##   LimeStatus indicating success or failure
  
  # First write with placeholder plaquette/linkTrace (all ranks participate)
  let tmpXml = makeGaugeUserXml(0.0, 0.0, userXml)
  result = writeGaugeField(u.field, filename, recordXml = tmpXml)
  
  if result != lsSuccess:
    return
  
  # Compute plaquette/linkTrace from the written file's raw binary data on rank 0.
  # This avoids the broken stencil and works on any number of ranks.
  var plaq, lt: float64
  if GA_Nodeid() == 0:
    let reader = newTensorFieldReader(filename)
    reader.loadBinaryData(validate = false)
    let nc = reader.colors
    let dims = reader.dims
    plaq = computePlaquetteFromBinary(reader.binaryData, dims, nc, swapBytes)
    lt = computeLinkTraceFromBinary(reader.binaryData, dims, nc, swapBytes)
    reader.close()
  
  GA_Sync()
  
  # Re-write with correct plaquette/linkTrace in recordXml (all ranks participate)
  let gaugeXml = makeGaugeUserXml(plaq, lt, userXml)
  result = writeGaugeField(u.field, filename, recordXml = gaugeXml)
