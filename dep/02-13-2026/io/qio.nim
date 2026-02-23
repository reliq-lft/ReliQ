#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/qio.nim
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

## QIO-style parallel I/O for lattice field data
##
## This module provides high-level parallel I/O facilities compatible with
## the QIO library format. It handles:
## - Parallel reading/writing with proper site ordering
## - Checksumming for data integrity
## - Byte swapping for portability
##
## Site ordering in QIO files follows lexicographic order:
##   ``site = x + Lx * (y + Ly * (z + Lz * t))``
##
## Reference: https://github.com/usqcd-software/qio

import std/[strformat]
import scidac
export scidac

# ===========================================================================
# Checksum utilities (QIO uses a specific checksum algorithm)
# ===========================================================================

type
  QIOChecksum* = object
    ## QIO uses a 32-bit checksum with two components
    suma*: uint32  # Sum of (rank XOR data)
    sumb*: uint32  # Sum of data

proc initChecksum*(): QIOChecksum =
  QIOChecksum(suma: 0, sumb: 0)

proc update*(cs: var QIOChecksum, data: openArray[byte], rank: uint32 = 0) =
  ## Update checksum with data
  for i in 0..<(data.len div 4):
    var word: uint32
    copyMem(addr word, unsafeAddr data[i * 4], 4)
    cs.suma = cs.suma xor (rank xor word)
    cs.sumb = cs.sumb + word

proc combine*(cs1, cs2: QIOChecksum): QIOChecksum =
  ## Combine two checksums (for parallel reduction)
  QIOChecksum(
    suma: cs1.suma xor cs2.suma,
    sumb: cs1.sumb + cs2.sumb
  )

proc `$`*(cs: QIOChecksum): string =
  fmt"({cs.suma:08x}, {cs.sumb:08x})"

# ===========================================================================
# Site ordering utilities
# ===========================================================================

proc lexIndex*(coords: openArray[int], dims: openArray[int]): int =
  ## Convert lattice coordinates to lexicographic index.
  ## QIO order: ``x + Lx * (y + Ly * (z + Lz * t))``
  result = 0
  var stride = 1
  for i in 0..<coords.len:
    result += coords[i] * stride
    stride *= dims[i]

proc lexCoords*(index: int, dims: openArray[int]): seq[int] =
  ## Convert lexicographic index to coordinates
  result = newSeq[int](dims.len)
  var idx = index
  for i in 0..<dims.len:
    result[i] = idx mod dims[i]
    idx = idx div dims[i]

proc evenOddIndex*(coords: openArray[int], dims: openArray[int]): int =
  ## Convert to even-odd (checkerboard) ordering
  ## Even sites first, then odd sites
  var sum = 0
  for c in coords: sum += c
  let isOdd = sum mod 2
  
  var vol = 1
  for d in dims: vol *= d
  let halfVol = vol div 2
  
  let lex = lexIndex(coords, dims)
  # Map to even-odd index
  if isOdd == 0:
    result = lex div 2
  else:
    result = halfVol + lex div 2

# ===========================================================================
# Byte swapping utilities
# ===========================================================================

type
  ByteOrder* = enum
    boNative
    boBigEndian
    boLittleEndian

proc needsSwap*(target: ByteOrder): bool =
  ## Check if byte swapping is needed
  when cpuEndian == littleEndian:
    result = (target == boBigEndian)
  else:
    result = (target == boLittleEndian)

proc swapBytes32*(data: var openArray[byte]) =
  ## Swap bytes for 32-bit values in-place
  var i = 0
  while i + 3 < data.len:
    swap(data[i], data[i + 3])
    swap(data[i + 1], data[i + 2])
    i += 4

proc swapBytes64*(data: var openArray[byte]) =
  ## Swap bytes for 64-bit values in-place
  var i = 0
  while i + 7 < data.len:
    swap(data[i], data[i + 7])
    swap(data[i + 1], data[i + 6])
    swap(data[i + 2], data[i + 5])
    swap(data[i + 3], data[i + 4])
    i += 8

proc swapBytesFloat32*(data: var openArray[float32]) =
  ## Swap bytes for float32 array
  var bytes = cast[ptr UncheckedArray[byte]](addr data[0])
  var byteSeq = newSeq[byte](data.len * 4)
  copyMem(addr byteSeq[0], bytes, byteSeq.len)
  swapBytes32(byteSeq)
  copyMem(bytes, addr byteSeq[0], byteSeq.len)

proc swapBytesFloat64*(data: var openArray[float64]) =
  ## Swap bytes for float64 array
  var bytes = cast[ptr UncheckedArray[byte]](addr data[0])
  var byteSeq = newSeq[byte](data.len * 8)
  copyMem(addr byteSeq[0], bytes, byteSeq.len)
  swapBytes64(byteSeq)
  copyMem(bytes, addr byteSeq[0], byteSeq.len)

# ===========================================================================
# QIO Field Reader
# ===========================================================================

type
  QIOFieldReader*[T] = ref object
    scidac: SciDACReader
    fileInfo*: SciDACFileInfo
    recordInfo*: SciDACRecordInfo
    volume: int
    siteSize: int  # bytes per site
    
proc newQIOFieldReader*[T](filename: string): QIOFieldReader[T] =
  ## Open a QIO file for reading field data
  result = QIOFieldReader[T](
    scidac: newSciDACReader(filename)
  )
  result.fileInfo = result.scidac.readFileInfo()
  result.volume = 1
  for d in result.fileInfo.dims:
    result.volume *= d

proc close*[T](r: QIOFieldReader[T]) =
  r.scidac.close()

proc dims*[T](r: QIOFieldReader[T]): seq[int] =
  r.fileInfo.dims

proc readField*[T](r: QIOFieldReader[T], 
                   targetOrder: ByteOrder = boBigEndian): seq[T] =
  ## Read a field from the file
  ## Data is returned in lexicographic site order
  
  for (recInfo, userXml, checksum) in r.scidac.messages:
    r.recordInfo = recInfo
    let data = r.scidac.readBinaryData()
    
    # Calculate elements per site
    let floatSize = if recInfo.precision == spSingle: 4 else: 8
    let bytesPerSite = recInfo.typesize * floatSize * recInfo.datacount
    r.siteSize = bytesPerSite
    
    let count = data.len div sizeof(T)
    result = newSeq[T](count)
    if count > 0:
      copyMem(addr result[0], unsafeAddr data[0], data.len)
    
    # Handle byte swapping if needed
    if needsSwap(targetOrder):
      when sizeof(T) == 4:
        var bytes = cast[ptr UncheckedArray[byte]](addr result[0])
        var byteSeq = newSeq[byte](result.len * sizeof(T))
        copyMem(addr byteSeq[0], bytes, byteSeq.len)
        swapBytes32(byteSeq)
        copyMem(bytes, addr byteSeq[0], byteSeq.len)
      elif sizeof(T) == 8:
        var bytes = cast[ptr UncheckedArray[byte]](addr result[0])
        var byteSeq = newSeq[byte](result.len * sizeof(T))
        copyMem(addr byteSeq[0], bytes, byteSeq.len)
        swapBytes64(byteSeq)
        copyMem(bytes, addr byteSeq[0], byteSeq.len)
    
    break  # Read first field only

# ===========================================================================
# QIO Field Writer
# ===========================================================================

type
  QIOFieldWriter*[T] = ref object
    scidac: SciDACWriter
    fileInfo*: SciDACFileInfo
    volume: int

proc newQIOFieldWriter*[T](filename: string, dims: openArray[int]): QIOFieldWriter[T] =
  ## Create a new QIO file for writing
  let fileInfo = newFileInfo(dims)
  result = QIOFieldWriter[T](
    scidac: newSciDACWriter(filename, fileInfo),
    fileInfo: fileInfo
  )
  result.volume = 1
  for d in dims:
    result.volume *= d

proc close*[T](w: QIOFieldWriter[T]): LimeStatus =
  w.scidac.close()

proc writeField*[T](w: QIOFieldWriter[T], 
                    data: openArray[T],
                    colors: int = 3,
                    spins: int = 0,
                    datacount: int = 4,
                    userXml: string = "",
                    sourceOrder: ByteOrder = boNative): LimeStatus =
  ## Write a field to the file
  ## 
  ## Parameters:
  ##   data: Field data in lexicographic site order
  ##   colors: Number of colors (default 3 for SU(3))
  ##   spins: Number of spin components (0 for gauge, 4 for fermions)
  ##   datacount: Number of data objects per site (4 for gauge links)
  ##   userXml: Optional user metadata XML
  ##   sourceOrder: Byte order of source data
  
  # Determine precision from type size
  let precision = if sizeof(T) == 4: spSingle else: spDouble
  let typesize = if spins == 0: colors * colors * 2 else: colors * spins * 2
  
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
    datacount: datacount
  )
  
  # Handle byte order conversion to big-endian (standard for QIO)
  if needsSwap(boBigEndian) and sourceOrder == boNative:
    var dataCopy = @data
    when sizeof(T) == 4:
      var bytes = cast[ptr UncheckedArray[byte]](addr dataCopy[0])
      var byteSeq = newSeq[byte](dataCopy.len * sizeof(T))
      copyMem(addr byteSeq[0], bytes, byteSeq.len)
      swapBytes32(byteSeq)
      copyMem(bytes, addr byteSeq[0], byteSeq.len)
    elif sizeof(T) == 8:
      var bytes = cast[ptr UncheckedArray[byte]](addr dataCopy[0])
      var byteSeq = newSeq[byte](dataCopy.len * sizeof(T))
      copyMem(addr byteSeq[0], bytes, byteSeq.len)
      swapBytes64(byteSeq)
      copyMem(bytes, addr byteSeq[0], byteSeq.len)
    result = w.scidac.writeField(dataCopy, recordInfo, userXml)
  else:
    result = w.scidac.writeField(data, recordInfo, userXml)

# ===========================================================================
# Convenience functions for common lattice QCD data types
# ===========================================================================

type
  QIOGaugeField*[T] = object
    ## Low-level QIO gauge field container (flat seq-based, non-distributed).
    ## For distributed gauge fields, use ``GaugeField`` from ``gauge/gaugefield``.
    dims*: array[4, int]
    data*: seq[T]  # Stored as real numbers

  QIOPropagatorField*[T] = object
    ## Low-level QIO propagator field container (flat seq-based, non-distributed).
    dims*: array[4, int]
    data*: seq[T]

proc volume*(dims: array[4, int]): int =
  dims[0] * dims[1] * dims[2] * dims[3]

proc readQIOGaugeField*(filename: string): QIOGaugeField[float64] =
  ## Read a gauge configuration from a QIO file into a flat (non-distributed) container.
  ## For distributed reads, use ``gaugeio.readGaugeField`` with a ``GaugeField``.
  let reader = newQIOFieldReader[float64](filename)
  defer: reader.close()
  
  let dims = reader.dims
  assert dims.len == 4, "Expected 4D lattice"
  
  result.dims = [dims[0], dims[1], dims[2], dims[3]]
  result.data = reader.readField[:float64]()

proc writeQIOGaugeField*(filename: string, field: QIOGaugeField[float64], 
                         userXml: string = ""): LimeStatus =
  ## Write a gauge configuration from a flat (non-distributed) container to a QIO file.
  ## For distributed writes, use ``gaugeio.writeGaugeField`` with a ``GaugeField``.
  let writer = newQIOFieldWriter[float64](filename, field.dims)
  defer: discard writer.close()
  
  result = writer.writeField(field.data, 
                              colors = 3,
                              spins = 0,
                              datacount = 4,
                              userXml = userXml)

when isMainModule:
  # Test lexicographic ordering
  let dims = [4, 4, 4, 8]
  echo "Testing site ordering..."
  
  let coords = @[1, 2, 3, 4]
  let idx = lexIndex(coords, dims)
  let back = lexCoords(idx, dims)
  echo fmt"  coords {coords} -> index {idx} -> coords {back}"
  assert back == coords
  
  echo "Testing checksum..."
  var cs = initChecksum()
  let testData = [byte 1, 2, 3, 4, 5, 6, 7, 8]
  cs.update(testData, 0)
  echo fmt"  Checksum: {cs}"
  
  echo "Testing QIO writer..."
  block:
    let vol = volume([4, 4, 4, 8])
    var data = newSeq[float64](vol * 4 * 18)  # 4 links * 18 reals per SU(3)
    for i in 0..<data.len:
      data[i] = float64(i) * 0.0001
    
    let field = QIOGaugeField[float64](
      dims: [4, 4, 4, 8],
      data: data
    )
    
    discard writeQIOGaugeField("/tmp/test_qio.lime", field, 
                               "<info>Test gauge configuration</info>")
    echo "  Written /tmp/test_qio.lime"
  
  echo "\nDone! Testing complete."
