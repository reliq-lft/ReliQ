#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/tensorfield.nim
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

## TensorField I/O facilities for reading/writing distributed tensor fields
##
## This module provides I/O operations for TensorField types, handling the 
## conversion between file formats (LIME/SciDAC/QIO) and distributed GlobalArray
## storage via LocalTensorField intermediaries.
##
## Key features:
## - Read LIME/SciDAC/QIO files into TensorField objects
## - Write TensorField objects to LIME/SciDAC/QIO files  
## - Handle parallel I/O with proper site ordering (rank 0 does file I/O)
## - Support for both real and complex tensor data
##
## Example usage:
## ```nim
## import io/tensorfield
## import tensor/[globaltensor, localtensor]
## import lattice
##
## let lattice = newSimpleCubicLattice([16, 16, 16, 32])
## var gaugeField = lattice.newTensorField([4, 3, 3]): Complex64
##
## # Read from file
## readTensorField(gaugeField, "config.lime")
##
## # Write to file
## writeTensorField(gaugeField, "output.lime", "<info>My config</info>")
## ```

import std/[math, strformat, endians]

import qio
import scidac
import lime

import ../lattice
import ../tensor/tensor

import ../globalarrays/[gatypes, gawrap]
import ../utils/[complex]

# ===========================================================================
# Site ordering utilities for TensorField I/O
# ===========================================================================

proc globalLexIndex*[D: static[int]](coords: array[D, int], dims: array[D, int]): int =
  ## Convert lattice coordinates to global lexicographic index
  ## QIO order: x + Lx*(y + Ly*(z + Lz*t))
  result = 0
  var stride = 1
  for i in 0..<D:
    result += coords[i] * stride
    stride *= dims[i]

proc globalLexCoords*[D: static[int]](index: int, dims: array[D, int]): array[D, int] =
  ## Convert global lexicographic index to coordinates
  var idx = index
  for i in 0..<D:
    result[i] = idx mod dims[i]
    idx = idx div dims[i]

# ===========================================================================
# TensorField Reader
# ===========================================================================

type
  TensorFieldReader* = ref object
    ## Reader for TensorField data from LIME/SciDAC/QIO files
    scidac: SciDACReader
    fileInfo*: SciDACFileInfo
    recordInfo*: SciDACRecordInfo
    storedChecksum*: SciDACChecksum
    computedChecksum*: SciDACChecksum
    binaryData: seq[byte]
    hasData: bool
    hasChecksum*: bool

proc newTensorFieldReader*(filename: string): TensorFieldReader =
  ## Open a file for reading tensor field data
  result = TensorFieldReader(
    scidac: newSciDACReader(filename),
    hasData: false,
    hasChecksum: false
  )
  result.fileInfo = result.scidac.readFileInfo()

proc close*(reader: TensorFieldReader) =
  ## Close the reader
  reader.scidac.close()

proc dims*(reader: TensorFieldReader): seq[int] =
  ## Get lattice dimensions from file
  reader.fileInfo.dims

proc printFileInfo*(reader: TensorFieldReader) =
  ## Print file metadata in a nice format
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║                    SciDAC/ILDG File Info                       ║"
  echo "╠════════════════════════════════════════════════════════════════╣"
  echo fmt"║  Version:     {reader.fileInfo.version:<48} ║"
  echo fmt"║  Spacetime:   {reader.fileInfo.spacetime:<48} ║"
  var dimStr = ""
  for i, d in reader.fileInfo.dims:
    if i > 0: dimStr.add " x "
    dimStr.add $d
  echo fmt"║  Dimensions:  {dimStr:<48} ║"
  echo fmt"║  Volume fmt:  {reader.fileInfo.volfmt:<48} ║"
  echo "╚════════════════════════════════════════════════════════════════╝"

proc printRecordInfo*(reader: TensorFieldReader) =
  ## Print record metadata in a nice format
  echo "╔════════════════════════════════════════════════════════════════╗"
  echo "║                    SciDAC Record Info                          ║"
  echo "╠════════════════════════════════════════════════════════════════╣"
  echo fmt"║  Datatype:    {reader.recordInfo.datatype:<48} ║"
  echo fmt"║  Precision:   {reader.recordInfo.precision:<48} ║"
  echo fmt"║  Colors:      {reader.recordInfo.colors:<48} ║"
  echo fmt"║  Spins:       {reader.recordInfo.spins:<48} ║"
  echo fmt"║  Type size:   {reader.recordInfo.typesize:<48} ║"
  echo fmt"║  Data count:  {reader.recordInfo.datacount:<48} ║"
  if reader.hasChecksum:
    echo "╠════════════════════════════════════════════════════════════════╣"
    let storedStr = fmt"suma={reader.storedChecksum.suma:08x}  sumb={reader.storedChecksum.sumb:08x}"
    let computedStr = fmt"suma={reader.computedChecksum.suma:08x}  sumb={reader.computedChecksum.sumb:08x}"
    echo fmt"║  Stored checksum:   {storedStr:<42} ║"
    echo fmt"║  Computed checksum: {computedStr:<42} ║"
    let match = reader.storedChecksum.suma == reader.computedChecksum.suma and
                reader.storedChecksum.sumb == reader.computedChecksum.sumb
    let status = if match: "PASSED ✓" else: "FAILED ✗"
    echo fmt"║  Checksum status:   {status:<42} ║"
  echo "╚════════════════════════════════════════════════════════════════╝"

proc loadBinaryData*(reader: TensorFieldReader, validate: bool = true) =
  ## Load binary data from the file (call before reading into TensorField)
  ## If validate is true and checksum doesn't match, raises IOError
  if not reader.hasData:
    for (recInfo, userXml, checksum) in reader.scidac.messages:
      reader.recordInfo = recInfo
      # Read binary data first - this also reads the checksum record that follows
      reader.binaryData = reader.scidac.readBinaryData()
      # Now get checksum info (updated by readBinaryData)
      reader.storedChecksum = reader.scidac.storedChecksum
      reader.hasChecksum = reader.scidac.hasChecksum
      reader.hasData = true
      
      # Compute checksum on the data we read using SciDAC/QIO algorithm
      # (CRC32 of each site's data, rotated by rank % 29 and rank % 31)
      reader.computedChecksum = initSciDACChecksum()
      
      # Calculate bytes per site
      var globalVol = 1
      for d in reader.fileInfo.dims:
        globalVol *= d
      let bytesPerSite = reader.binaryData.len div globalVol
      
      # Compute checksum over all sites using the proper algorithm
      for site in 0..<globalVol:
        let offset = site * bytesPerSite
        let siteData = reader.binaryData[offset ..< offset + bytesPerSite]
        reader.computedChecksum.updateChecksum(siteData, uint32(site))
      
      # Validate checksum if requested
      if validate and reader.hasChecksum:
        if reader.storedChecksum.suma != reader.computedChecksum.suma or
           reader.storedChecksum.sumb != reader.computedChecksum.sumb:
          raise newException(IOError, 
            fmt"Checksum validation failed! Stored: {reader.storedChecksum}, Computed: {reader.computedChecksum}")
      
      break  # Read first field only

proc precision*(reader: TensorFieldReader): SciDACPrecision =
  ## Get the precision of the data in the file
  reader.recordInfo.precision

proc colors*(reader: TensorFieldReader): int =
  ## Get number of colors (e.g., 3 for SU(3))
  reader.recordInfo.colors

proc spins*(reader: TensorFieldReader): int =
  ## Get number of spin components (0 for gauge, 4 for fermions)
  reader.recordInfo.spins

proc typesize*(reader: TensorFieldReader): int =
  ## Get type size (number of real values per site element)
  reader.recordInfo.typesize

proc datacount*(reader: TensorFieldReader): int =
  ## Get data count (number of tensor elements per site)
  reader.recordInfo.datacount

# ===========================================================================
# Read TensorField from file
# ===========================================================================

proc readTensorField*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: var TensorField[D, R, L, T],
  filename: string,
  swapBytes: bool = true
) =
  ## Read tensor field data from a LIME/SciDAC/QIO file
  ##
  ## The file data is read by rank 0 and distributed to all ranks via the
  ## GlobalArray. Each rank receives the portion of data it owns.
  ##
  ## Parameters:
  ##   tensor: The TensorField to read data into (must be pre-allocated)
  ##   filename: Path to the input file
  ##   swapBytes: Whether to swap bytes for endianness (default: true for big-endian files)
  
  # Synchronize before starting
  GA_Sync()
  
  # Calculate element sizes
  var elemsPerSite = 1
  for i in 0..<R:
    elemsPerSite *= tensor.shape[i]
  
  # For complex types, each element is 2 floats (re, im)
  let realsPerSite = when isComplex(T): elemsPerSite * 2 else: elemsPerSite
  
  # Determine precision (assume double for now, can be read from file)
  let precision = when T is float32 or isComplex32(T): spSingle else: spDouble
  let floatSize = if precision == spSingle: 4 else: 8
  let bytesPerSite = realsPerSite * floatSize
  
  # Calculate global volume
  var globalVol = 1
  for i in 0..<D:
    globalVol *= tensor.lattice.globalGrid[i]
  
  # Only rank 0 reads the file
  var data: seq[byte]
  if GA_Nodeid() == 0:
    let reader = newTensorFieldReader(filename)
    defer: reader.close()
    
    # Verify dimensions match
    let fileDims = reader.dims
    assert fileDims.len == D, fmt"File has {fileDims.len}D lattice, expected {D}D"
    for i in 0..<D:
      assert tensor.lattice.globalGrid[i] == fileDims[i],
        fmt"Dimension mismatch: lattice[{i}]={tensor.lattice.globalGrid[i]}, file[{i}]={fileDims[i]}"
    
    reader.loadBinaryData()
    data = reader.binaryData
    
    # Handle byte swapping if needed (files are typically big-endian)
    if swapBytes and needsSwap(boBigEndian):
      if floatSize == 4:
        swapBytes32(data)
      else:
        swapBytes64(data)
  
  # Broadcast file data from rank 0 to all ranks using GlobalArrays
  # Create a temporary GlobalArray to hold the file data
  GA_Sync()
  
  # Get local tensor view for writing
  var localTensor = tensor.newLocalTensorField()
  
  # Get bounds for this rank's portion
  let handle = tensor.data.getHandle()
  var lo, hi: array[D + R + 1, cint]
  let pid = GA_Nodeid()
  handle.NGA_Distribution(pid, addr lo[0], addr hi[0])
  
  # Extract lattice bounds (first D dimensions)
  var latticeLo, latticeHi: array[D, int]
  for i in 0..<D:
    latticeLo[i] = int(lo[i])
    latticeHi[i] = int(hi[i])
  
  # Use GA_Brdcst to broadcast the data from rank 0 to all ranks
  var dataLen = globalVol * bytesPerSite
  if GA_Nodeid() != 0:
    data = newSeq[byte](dataLen)
  
  GA_Brdcst(addr data[0], cint(dataLen), cint(0))
  
  # Now each rank has the full file data and can extract its local portion
  let numLocalSites = localTensor.numGlobalSites()
  
  for localSite in 0..<numLocalSites:
    # Convert local site index to local coordinates
    var localCoords: array[D, int]
    var idx = localSite
    for i in 0..<D:
      localCoords[i] = idx mod localTensor.localGrid[i]
      idx = idx div localTensor.localGrid[i]
    
    # Convert to global coordinates
    var globalCoords: array[D, int]
    for i in 0..<D:
      globalCoords[i] = latticeLo[i] + localCoords[i]
    
    # Calculate global lexicographic index
    let globalLex = globalLexIndex(globalCoords, tensor.lattice.globalGrid)
    
    # Calculate byte offset in file data
    let fileOffset = globalLex * bytesPerSite
    
    # Copy data to local tensor
    let localOffset = localSite * realsPerSite
    
    when isComplex64(T):
      for e in 0..<elemsPerSite:
        var re, im: float64
        copyMem(addr re, unsafeAddr data[fileOffset + e * 16], 8)
        copyMem(addr im, unsafeAddr data[fileOffset + e * 16 + 8], 8)
        localTensor.data[localOffset + e * 2] = re
        localTensor.data[localOffset + e * 2 + 1] = im
    elif isComplex32(T):
      for e in 0..<elemsPerSite:
        var re, im: float32
        copyMem(addr re, unsafeAddr data[fileOffset + e * 8], 4)
        copyMem(addr im, unsafeAddr data[fileOffset + e * 8 + 4], 4)
        localTensor.data[localOffset + e * 2] = re
        localTensor.data[localOffset + e * 2 + 1] = im
    elif T is float64:
      for e in 0..<elemsPerSite:
        var val: float64
        copyMem(addr val, unsafeAddr data[fileOffset + e * 8], 8)
        localTensor.data[localOffset + e] = val
    elif T is float32:
      for e in 0..<elemsPerSite:
        var val: float32
        copyMem(addr val, unsafeAddr data[fileOffset + e * 4], 4)
        localTensor.data[localOffset + e] = val
  
  # Synchronize global array after writing local portion
  GA_Sync()

# ===========================================================================
# TensorField Writer
# ===========================================================================

type
  TensorFieldWriter* = ref object
    ## Writer for TensorField data to LIME/SciDAC/QIO files
    scidac: SciDACWriter
    fileInfo*: SciDACFileInfo
    wroteHeader: bool

proc newTensorFieldWriter*[D: static[int]](
  filename: string,
  dims: array[D, int]
): TensorFieldWriter =
  ## Create a new writer for tensor field data
  var dimSeq = newSeq[int](D)
  for i in 0..<D:
    dimSeq[i] = dims[i]
  
  let fileInfo = SciDACFileInfo(
    version: "1.0",
    spacetime: D,
    dims: dimSeq,
    volfmt: 0
  )
  
  result = TensorFieldWriter(
    scidac: newSciDACWriter(filename, fileInfo),
    fileInfo: fileInfo,
    wroteHeader: false
  )

proc close*(writer: TensorFieldWriter): LimeStatus =
  ## Close the writer
  writer.scidac.close()

proc writeTensorField*[D: static[int], R: static[int], L: Lattice[D], T](
  tensor: TensorField[D, R, L, T],
  filename: string,
  userXml: string = "",
  colors: int = 3,
  spins: int = 0
): LimeStatus =
  ## Write tensor field data to a LIME/SciDAC/QIO file
  ##
  ## The distributed TensorField data is gathered to rank 0 using GlobalArrays
  ## broadcast mechanism (in reverse - we gather). Rank 0 writes the file.
  ##
  ## Parameters:
  ##   tensor: The TensorField to write
  ##   filename: Path to the output file
  ##   userXml: Optional user metadata XML
  ##   colors: Number of colors (default 3 for SU(3))
  ##   spins: Number of spin components (0 for gauge, 4 for fermions)
  ##
  ## Returns:
  ##   LimeStatus indicating success or failure
  
  # Synchronize before starting
  GA_Sync()
  
  # Calculate elements per site from tensor shape
  var elemsPerSite = 1
  for i in 0..<R:
    elemsPerSite *= tensor.shape[i]
  
  # For complex types, each element is 2 floats (re, im)
  let realsPerSite = when isComplex(T): elemsPerSite * 2 else: elemsPerSite
  
  # Calculate global volume
  var globalVol = 1
  for i in 0..<D:
    globalVol *= tensor.lattice.globalGrid[i]
  
  # Determine precision and float size
  let precision = when T is float32 or isComplex32(T): spSingle else: spDouble
  let floatSize = if precision == spSingle: 4 else: 8
  let bytesPerSite = realsPerSite * floatSize
  
  # Get local tensor view
  var localTensor = tensor.newLocalTensorField()
  
  # Get bounds for this rank's portion
  let handle = tensor.data.getHandle()
  var lo, hi: array[D + R + 1, cint]
  let pid = GA_Nodeid()
  handle.NGA_Distribution(pid, addr lo[0], addr hi[0])
  
  # Extract lattice bounds (first D dimensions)
  var latticeLo, latticeHi: array[D, int]
  for i in 0..<D:
    latticeLo[i] = int(lo[i])
    latticeHi[i] = int(hi[i])
  
  # Allocate global data buffer on all ranks (for gathering via GA operations)
  var globalData = newSeq[byte](globalVol * bytesPerSite)
  
  # Each rank fills in its portion of globalData at the correct positions
  let numLocalSites = localTensor.numGlobalSites()
  
  for localSite in 0..<numLocalSites:
    # Convert local site index to local coordinates
    var localCoords: array[D, int]
    var idx = localSite
    for i in 0..<D:
      localCoords[i] = idx mod localTensor.localGrid[i]
      idx = idx div localTensor.localGrid[i]
    
    # Convert to global coordinates
    var globalCoords: array[D, int]
    for i in 0..<D:
      globalCoords[i] = latticeLo[i] + localCoords[i]
    
    # Calculate global lexicographic index
    let globalLex = globalLexIndex(globalCoords, tensor.lattice.globalGrid)
    
    # Calculate byte offset in global data
    let fileOffset = globalLex * bytesPerSite
    
    # Copy data from local tensor to global buffer
    let localOffset = localSite * realsPerSite
    
    when isComplex64(T):
      for e in 0..<elemsPerSite:
        let re = localTensor.data[localOffset + e * 2]
        let im = localTensor.data[localOffset + e * 2 + 1]
        copyMem(addr globalData[fileOffset + e * 16], unsafeAddr re, 8)
        copyMem(addr globalData[fileOffset + e * 16 + 8], unsafeAddr im, 8)
    elif isComplex32(T):
      for e in 0..<elemsPerSite:
        let re = localTensor.data[localOffset + e * 2]
        let im = localTensor.data[localOffset + e * 2 + 1]
        copyMem(addr globalData[fileOffset + e * 8], unsafeAddr re, 4)
        copyMem(addr globalData[fileOffset + e * 8 + 4], unsafeAddr im, 4)
    elif T is float64:
      for e in 0..<elemsPerSite:
        let val = localTensor.data[localOffset + e]
        copyMem(addr globalData[fileOffset + e * 8], unsafeAddr val, 8)
    elif T is float32:
      for e in 0..<elemsPerSite:
        let val = localTensor.data[localOffset + e]
        copyMem(addr globalData[fileOffset + e * 4], unsafeAddr val, 4)
  
  # Use GA_Dgop to sum the data across all ranks
  # Each rank has zeros where it doesn't own data, so sum gives the full field
  # Note: We're using bytes as a buffer, so treat as integer sum operation
  # Actually, GA_Dgop expects double, so we need to be careful
  # Use GA_Igop for integer data (treating bytes as ints)
  let numInts = (globalData.len + sizeof(cint) - 1) div sizeof(cint)
  var intData = newSeq[cint](numInts)
  copyMem(addr intData[0], addr globalData[0], globalData.len)
  
  # Zero out areas this rank doesn't own before the sum
  # Actually we need a different approach - use GA_Gop with "+" operation
  GA_Igop(addr intData[0], cint(numInts), cstring("+"))
  
  copyMem(addr globalData[0], addr intData[0], globalData.len)
  
  # Now rank 0 has the complete field (all ranks do, actually)
  if GA_Nodeid() == 0:
    # Swap bytes to big-endian for file format
    if needsSwap(boBigEndian):
      if floatSize == 4:
        swapBytes32(globalData)
      else:
        swapBytes64(globalData)
    
    # Write the file
    let writer = newTensorFieldWriter(filename, tensor.lattice.globalGrid)
    defer: discard writer.close()
    
    # Write file header
    discard writer.scidac.writeFileInfo(userXml)
    
    # Create record info
    let typesize = if spins == 0: colors * colors * 2 else: colors * spins * 2
    let datacount = elemsPerSite div (colors * colors)  # Assuming gauge-like structure
    
    let recordInfo = SciDACRecordInfo(
      version: "1.0",
      date: "",
      recordtype: 0,
      datatype: if spins == 0:
                  fmt"QDP_{precision}{colors}_ColorMatrix"
                else:
                  fmt"QDP_{precision}{colors}_DiracFermion",
      precision: precision,
      colors: colors,
      spins: spins,
      typesize: typesize,
      datacount: if datacount > 0: datacount else: 1
    )
    
    # Write the binary data
    result = writer.scidac.writeField(globalData, recordInfo, "")
  else:
    result = lsSuccess
  
  GA_Sync()

# ===========================================================================
# Convenience functions for common field types
# ===========================================================================

proc readGaugeField*[D: static[int], L: Lattice[D]](
  gaugeField: var array[4, TensorField[D, 2, L, Complex64]],
  filename: string,
  swapBytes: bool = true
) =
  ## Read a gauge configuration into an array of 4 TensorFields with shape [3, 3]
  ## 
  ## The gauge field has 4 link matrices per site (one per direction),
  ## each being a 3x3 complex SU(3) matrix stored as a separate TensorField.
  ##
  ## Parameters:
  ##   gaugeField: Array of 4 TensorFields, each with shape [3, 3]
  ##   filename: Path to the input file
  ##   swapBytes: Whether to swap bytes for endianness (default: true)
  
  GA_Sync()
  
  # Each direction has 9 complex elements = 18 floats
  const elemsPerDir = 9
  const realsPerDir = elemsPerDir * 2  # Complex64 = 2 floats
  const floatSize = 8  # double precision
  const bytesPerDir = realsPerDir * floatSize
  const bytesPerSite = 4 * bytesPerDir  # 4 directions
  
  # Calculate global volume from the first tensor field
  var globalVol = 1
  for i in 0..<D:
    globalVol *= gaugeField[0].lattice.globalGrid[i]
  
  # Only rank 0 reads the file and prints metadata
  var data: seq[byte]
  if GA_Nodeid() == 0:
    let reader = newTensorFieldReader(filename)
    defer: reader.close()
    
    # Print file metadata
    reader.printFileInfo()
    
    # Verify dimensions match
    let fileDims = reader.dims
    assert fileDims.len == D, fmt"File has {fileDims.len}D lattice, expected {D}D"
    for i in 0..<D:
      assert gaugeField[0].lattice.globalGrid[i] == fileDims[i],
        fmt"Dimension mismatch: lattice[{i}]={gaugeField[0].lattice.globalGrid[i]}, file[{i}]={fileDims[i]}"
    
    # Load and validate data (checksum validation happens inside)
    reader.loadBinaryData(validate = true)
    
    # Print record info including checksum status
    reader.printRecordInfo()
    
    data = reader.binaryData
    
    # Handle byte swapping if needed (files are typically big-endian)
    if swapBytes and needsSwap(boBigEndian):
      swapBytes64(data)
  
  GA_Sync()
  
  # Broadcast file data from rank 0 to all ranks
  var dataLen = globalVol * bytesPerSite
  if GA_Nodeid() != 0:
    data = newSeq[byte](dataLen)
  
  GA_Brdcst(addr data[0], cint(dataLen), cint(0))
  
  # Each rank extracts its local portion for each direction
  for mu in 0..<4:
    var localTensor = gaugeField[mu].newLocalTensorField()
    
    # Get bounds for this rank's portion
    let handle = gaugeField[mu].data.getHandle()
    var lo, hi: array[D + 2 + 1, cint]
    let pid = GA_Nodeid()
    handle.NGA_Distribution(pid, addr lo[0], addr hi[0])
    
    # Extract lattice bounds (first D dimensions)
    var latticeLo: array[D, int]
    for i in 0..<D:
      latticeLo[i] = int(lo[i])
    
    let numLocalSites = localTensor.numSites()
    
    for localSite in 0..<numLocalSites:
      # Convert local site index to local coordinates
      var localCoords: array[D, int]
      var idx = localSite
      for i in 0..<D:
        localCoords[i] = idx mod localTensor.localGrid[i]
        idx = idx div localTensor.localGrid[i]
      
      # Convert to global coordinates
      var globalCoords: array[D, int]
      for i in 0..<D:
        globalCoords[i] = latticeLo[i] + localCoords[i]
      
      # Calculate global lexicographic index
      let globalLex = globalLexIndex(globalCoords, gaugeField[0].lattice.globalGrid)
      
      # Calculate byte offset in file data: site * (4 dirs * 18 reals * 8 bytes) + dir * (18 reals * 8 bytes)
      let siteOffset = globalLex * bytesPerSite
      let dirOffset = mu * bytesPerDir
      let fileOffset = siteOffset + dirOffset
      
      # Copy 9 complex elements (18 floats) to local tensor
      let localOffset = localSite * realsPerDir
      
      for e in 0..<elemsPerDir:
        var re, im: float64
        copyMem(addr re, unsafeAddr data[fileOffset + e * 16], 8)
        copyMem(addr im, unsafeAddr data[fileOffset + e * 16 + 8], 8)
        localTensor.data[localOffset + e * 2] = re
        localTensor.data[localOffset + e * 2 + 1] = im
  
  GA_Sync()

proc writeGaugeField*[D: static[int], L: Lattice[D]](
  gaugeField: array[4, TensorField[D, 2, L, Complex64]],
  filename: string,
  userXml: string = ""
): LimeStatus =
  ## Write a gauge configuration from an array of 4 TensorFields to file
  ##
  ## Parameters:
  ##   gaugeField: Array of 4 TensorFields, each with shape [3, 3]
  ##   filename: Path to the output file
  ##   userXml: Optional user metadata XML
  
  GA_Sync()
  
  const elemsPerDir = 9
  const realsPerDir = elemsPerDir * 2
  const floatSize = 8
  const bytesPerDir = realsPerDir * floatSize
  const bytesPerSite = 4 * bytesPerDir
  
  var globalVol = 1
  for i in 0..<D:
    globalVol *= gaugeField[0].lattice.globalGrid[i]
  
  # Allocate global data buffer on all ranks
  var globalData = newSeq[byte](globalVol * bytesPerSite)
  
  # Each rank fills in its portion for each direction
  for mu in 0..<4:
    var localTensor = gaugeField[mu].newLocalTensorField()
    
    let handle = gaugeField[mu].data.getHandle()
    var lo, hi: array[D + 2 + 1, cint]
    let pid = GA_Nodeid()
    handle.NGA_Distribution(pid, addr lo[0], addr hi[0])
    
    var latticeLo: array[D, int]
    for i in 0..<D:
      latticeLo[i] = int(lo[i])
    
    let numLocalSites = localTensor.numSites()
    
    for localSite in 0..<numLocalSites:
      var localCoords: array[D, int]
      var idx = localSite
      for i in 0..<D:
        localCoords[i] = idx mod localTensor.localGrid[i]
        idx = idx div localTensor.localGrid[i]
      
      var globalCoords: array[D, int]
      for i in 0..<D:
        globalCoords[i] = latticeLo[i] + localCoords[i]
      
      let globalLex = globalLexIndex(globalCoords, gaugeField[0].lattice.globalGrid)
      
      let siteOffset = globalLex * bytesPerSite
      let dirOffset = mu * bytesPerDir
      let fileOffset = siteOffset + dirOffset
      
      let localOffset = localSite * realsPerDir
      
      for e in 0..<elemsPerDir:
        let re = localTensor.data[localOffset + e * 2]
        let im = localTensor.data[localOffset + e * 2 + 1]
        copyMem(addr globalData[fileOffset + e * 16], unsafeAddr re, 8)
        copyMem(addr globalData[fileOffset + e * 16 + 8], unsafeAddr im, 8)
  
  # Gather data to all ranks using GA_Igop
  let numInts = (globalData.len + sizeof(cint) - 1) div sizeof(cint)
  var intData = newSeq[cint](numInts)
  copyMem(addr intData[0], addr globalData[0], globalData.len)
  GA_Igop(addr intData[0], cint(numInts), cstring("+"))
  copyMem(addr globalData[0], addr intData[0], globalData.len)
  
  # Rank 0 writes the file
  if GA_Nodeid() == 0:
    if needsSwap(boBigEndian):
      swapBytes64(globalData)
    
    let writer = newTensorFieldWriter(filename, gaugeField[0].lattice.globalGrid)
    defer: discard writer.close()
    
    discard writer.scidac.writeFileInfo(userXml)
    
    let recordInfo = SciDACRecordInfo(
      version: "1.0",
      date: "",
      recordtype: 0,
      datatype: "QDP_D3_ColorMatrix",
      precision: spDouble,
      colors: 3,
      spins: 0,
      typesize: 18,
      datacount: 4
    )
    
    result = writer.scidac.writeField(globalData, recordInfo, "")
  else:
    result = lsSuccess
  
  GA_Sync()

proc readGaugeFieldToTensor*[D: static[int], L: Lattice[D]](
  tensor: var TensorField[D, 3, L, Complex64],
  filename: string
) =
  ## Read a gauge configuration into a TensorField with shape [4, 3, 3]
  ## 
  ## The gauge field has 4 link matrices per site (one per direction),
  ## each being a 3x3 complex SU(3) matrix.
  assert tensor.shape == [4, 3, 3], "Gauge field must have shape [4, 3, 3]"
  readTensorField(tensor, filename)

proc writeGaugeFieldFromTensor*[D: static[int], L: Lattice[D]](
  tensor: TensorField[D, 3, L, Complex64],
  filename: string,
  userXml: string = ""
): LimeStatus =
  ## Write a gauge configuration from a TensorField to file
  assert tensor.shape == [4, 3, 3], "Gauge field must have shape [4, 3, 3]"
  writeTensorField(tensor, filename, userXml, colors = 3, spins = 0)

proc readPropagatorToTensor*[D: static[int], L: Lattice[D]](
  tensor: var TensorField[D, 4, L, Complex64],
  filename: string
) =
  ## Read a propagator field into a TensorField with shape [4, 3, 4, 3]
  ##
  ## The propagator has spin (4) and color (3) indices for both source and sink.
  assert tensor.shape == [4, 3, 4, 3], "Propagator field must have shape [4, 3, 4, 3]"
  readTensorField(tensor, filename)

proc writePropagatorFromTensor*[D: static[int], L: Lattice[D]](
  tensor: TensorField[D, 4, L, Complex64],
  filename: string,
  userXml: string = ""
): LimeStatus =
  ## Write a propagator field from a TensorField to file
  assert tensor.shape == [4, 3, 4, 3], "Propagator field must have shape [4, 3, 4, 3]"
  writeTensorField(tensor, filename, userXml, colors = 3, spins = 4)
