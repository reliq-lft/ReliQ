#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/scidac.nim
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

## SciDAC file format implementation built on LIME
##
## SciDAC files use LIME as the container format with specific record types
## and XML metadata for lattice field data. This module provides high-level
## abstractions for reading and writing SciDAC-formatted lattice data.
##
## Standard SciDAC record types:
## - scidac-private-file-xml: File-level metadata
## - scidac-private-record-xml: Record-level metadata  
## - scidac-binary-data: Binary field data
## - ildg-format: ILDG-specific format info
## - ildg-binary-data: ILDG gauge configuration data
##
## Reference: QIO library (https://github.com/usqcd-software/qio)

import std/[xmlparser, xmltree, strutils, strformat, tables, times]
import lime

export lime

# ===========================================================================
# CRC32 implementation (IEEE 802.3 polynomial)
# ===========================================================================

const crc32Table: array[256, uint32] = block:
  var table: array[256, uint32]
  const polynomial = 0xEDB88320'u32
  for i in 0..<256:
    var crc = uint32(i)
    for _ in 0..<8:
      if (crc and 1) != 0:
        crc = (crc shr 1) xor polynomial
      else:
        crc = crc shr 1
    table[i] = crc
  table

proc crc32*(data: openArray[byte]): uint32 =
  ## Compute CRC32 checksum (IEEE 802.3 polynomial)
  result = 0xFFFFFFFF'u32
  for b in data:
    let idx = (result xor uint32(b)) and 0xFF
    result = (result shr 8) xor crc32Table[idx]
  result = result xor 0xFFFFFFFF'u32

# ===========================================================================
# SciDAC record type identifiers
# ===========================================================================

const
  # Standard SciDAC/QIO record types
  SciDACPrivateFileXml* = "scidac-private-file-xml"
  SciDACFileXml* = "scidac-file-xml"
  SciDACPrivateRecordXml* = "scidac-private-record-xml"  
  SciDACRecordXml* = "scidac-record-xml"
  SciDACBinaryData* = "scidac-binary-data"
  SciDACChecksumRecord* = "scidac-checksum"
  
  # ILDG (International Lattice Data Grid) record types
  ILDGFormat* = "ildg-format"
  ILDGBinaryData* = "ildg-binary-data"
  ILDGDataLFN* = "ildg-data-lfn"

# ===========================================================================
# SciDAC metadata structures
# ===========================================================================

type
  SciDACPrecision* = enum
    spSingle = "F"
    spDouble = "D"

  SciDACTypeSize* = enum
    ## Common type sizes for lattice QCD
    stReal = 1           # Real number
    stComplex = 2        # Complex number
    stColorVector = 6    # SU(3) color vector (3 complex)
    stColorMatrix = 18   # SU(3) color matrix (3x3 complex)
    stDiracSpinor = 24   # Dirac spinor (4 color vectors)
    stPropagator = 288   # Full propagator (12x12 complex)

  SciDACChecksum* = object
    ## SciDAC checksum with two 32-bit components
    version*: string
    suma*: uint32  # XOR-based checksum
    sumb*: uint32  # Additive checksum

  SciDACFileInfo* = object
    ## Private file-level metadata
    version*: string
    spacetime*: int          # Number of spacetime dimensions
    dims*: seq[int]          # Lattice dimensions [x, y, z, t]
    volfmt*: int             # Volume format (0=single file, 1=multi-file)
    
  SciDACRecordInfo* = object
    ## Private record-level metadata
    version*: string
    date*: string
    recordtype*: int         # 0=field, 1=global
    datatype*: string        # e.g., "QDP_F3_ColorMatrix"
    precision*: SciDACPrecision
    colors*: int
    spins*: int
    typesize*: int           # Number of real numbers per site
    datacount*: int          # Number of data values per site

  SciDACUserInfo* = object
    ## User-supplied metadata (application-specific)
    info*: string

  ILDGFormatInfo* = object
    ## ILDG format metadata
    version*: string
    field*: string           # e.g., "su3gauge"  
    precision*: int          # 32 or 64
    lx*, ly*, lz*, lt*: int

proc `$`*(cs: SciDACChecksum): string =
  ## Format checksum as string
  fmt"suma={cs.suma:08x} sumb={cs.sumb:08x}"

proc initSciDACChecksum*(): SciDACChecksum =
  ## Initialize an empty checksum
  SciDACChecksum(version: "1.0", suma: 0, sumb: 0)

proc rotl32(value: uint32, shift: uint32): uint32 {.inline.} =
  ## Rotate left a 32-bit value by shift bits
  let mask = 31'u32  # 32 - 1
  let s = shift and mask
  if s == 0:
    value
  else:
    (value shl s) or (value shr (32'u32 - s))

proc updateChecksum*(cs: var SciDACChecksum, data: openArray[byte], siteIndex: uint32 = 0) =
  ## Update checksum with data from a site using SciDAC/QIO algorithm
  ## siteIndex is the global lexicographic site index
  ## Algorithm: CRC32 of site data, then rotated by (rank % 29) and (rank % 31)
  let work = crc32(data)
  let rank29 = siteIndex mod 29
  let rank31 = siteIndex mod 31
  cs.suma = cs.suma xor rotl32(work, rank29)
  cs.sumb = cs.sumb xor rotl32(work, rank31)

proc combineChecksums*(cs1, cs2: SciDACChecksum): SciDACChecksum =
  ## Combine two checksums (for parallel reduction)
  SciDACChecksum(
    version: cs1.version,
    suma: cs1.suma xor cs2.suma,
    sumb: cs1.sumb xor cs2.sumb  # Note: XOR, not addition
  )

proc parseChecksumXml*(xml: string): SciDACChecksum =
  ## Parse scidac-checksum XML record
  let tree = parseXml(xml)
  result.version = tree.child("version").innerText
  result.suma = uint32(parseHexInt(tree.child("suma").innerText))
  result.sumb = uint32(parseHexInt(tree.child("sumb").innerText))

proc generateChecksumXml*(cs: SciDACChecksum): string =
  ## Generate scidac-checksum XML record
  result = fmt"""<?xml version="1.0" encoding="UTF-8"?><scidacChecksum><version>{cs.version}</version><suma>{cs.suma:08x}</suma><sumb>{cs.sumb:08x}</sumb></scidacChecksum>"""

# ===========================================================================
# XML generation and parsing
# ===========================================================================

proc generateFileXml*(info: SciDACFileInfo): string =
  ## Generate SciDAC private file XML
  var dims = ""
  for d in info.dims:
    if dims.len > 0: dims.add " "
    dims.add $d
    
  result = fmt"""<?xml version="1.0" encoding="UTF-8"?>
<scidacFile>
  <version>{info.version}</version>
  <spacetime>{info.spacetime}</spacetime>
  <dims>{dims}</dims>
  <volfmt>{info.volfmt}</volfmt>
</scidacFile>
"""

proc generateRecordXml*(info: SciDACRecordInfo): string =
  ## Generate SciDAC private record XML
  result = fmt"""<?xml version="1.0" encoding="UTF-8"?>
<scidacRecord>
  <version>{info.version}</version>
  <date>{info.date}</date>
  <recordtype>{info.recordtype}</recordtype>
  <datatype>{info.datatype}</datatype>
  <precision>{info.precision}</precision>
  <colors>{info.colors}</colors>
  <spins>{info.spins}</spins>
  <typesize>{info.typesize}</typesize>
  <datacount>{info.datacount}</datacount>
</scidacRecord>
"""

proc generateUserXml*(info: SciDACUserInfo): string =
  ## Generate user info XML wrapper
  result = fmt"""<?xml version="1.0" encoding="UTF-8"?>
<info>{info.info}</info>
"""

proc generateILDGFormatXml*(info: ILDGFormatInfo): string =
  ## Generate ILDG format XML
  result = fmt"""<?xml version="1.0" encoding="UTF-8"?>
<ildgFormat>
  <version>{info.version}</version>
  <field>{info.field}</field>
  <precision>{info.precision}</precision>
  <lx>{info.lx}</lx>
  <ly>{info.ly}</ly>
  <lz>{info.lz}</lz>
  <lt>{info.lt}</lt>
</ildgFormat>
"""

proc parseFileXml*(xml: string): SciDACFileInfo =
  ## Parse SciDAC private file XML
  let tree = parseXml(xml)
  result.version = tree.child("version").innerText
  result.spacetime = parseInt(tree.child("spacetime").innerText)
  result.volfmt = parseInt(tree.child("volfmt").innerText)
  
  let dimsStr = tree.child("dims").innerText.strip()
  for d in dimsStr.splitWhitespace():
    result.dims.add parseInt(d)

proc parseRecordXml*(xml: string): SciDACRecordInfo =
  ## Parse SciDAC private record XML
  let tree = parseXml(xml)
  result.version = tree.child("version").innerText
  result.date = tree.child("date").innerText
  result.recordtype = parseInt(tree.child("recordtype").innerText)
  result.datatype = tree.child("datatype").innerText
  result.precision = if tree.child("precision").innerText == "F": spSingle else: spDouble
  result.colors = parseInt(tree.child("colors").innerText)
  result.spins = parseInt(tree.child("spins").innerText)
  result.typesize = parseInt(tree.child("typesize").innerText)
  result.datacount = parseInt(tree.child("datacount").innerText)

proc parseILDGFormatXml*(xml: string): ILDGFormatInfo =
  ## Parse ILDG format XML
  let tree = parseXml(xml)
  result.version = tree.child("version").innerText
  result.field = tree.child("field").innerText
  result.precision = parseInt(tree.child("precision").innerText)
  result.lx = parseInt(tree.child("lx").innerText)
  result.ly = parseInt(tree.child("ly").innerText)
  result.lz = parseInt(tree.child("lz").innerText)
  result.lt = parseInt(tree.child("lt").innerText)

# ===========================================================================
# High-level SciDAC reader
# ===========================================================================

type
  SciDACReader* = ref object
    reader*: LimeReader
    fileInfo*: SciDACFileInfo
    storedChecksum*: SciDACChecksum
    hasFileInfo: bool
    hasChecksum*: bool

proc newSciDACReader*(filename: string): SciDACReader =
  ## Open a SciDAC file for reading
  result = SciDACReader(
    reader: newLimeReader(filename),
    hasFileInfo: false,
    hasChecksum: false
  )

proc close*(r: SciDACReader) =
  r.reader.close()

proc readFileInfo*(r: SciDACReader): SciDACFileInfo =
  ## Read the file-level metadata (usually first record)
  if r.hasFileInfo:
    return r.fileInfo
    
  # Search for file info record
  while r.reader.nextRecord() == lsSuccess:
    if r.reader.limeType == SciDACPrivateFileXml:
      let xml = r.reader.readString()
      r.fileInfo = parseFileXml(xml)
      r.hasFileInfo = true
      return r.fileInfo
  
  raise newException(IOError, "No SciDAC file info found")

iterator messages*(r: SciDACReader): tuple[recordInfo: SciDACRecordInfo, userXml: string, checksum: SciDACChecksum] =
  ## Iterate over SciDAC messages, yielding record info, user XML, and checksum
  ## After yielding, the reader is positioned at the binary data record
  var recordInfo: SciDACRecordInfo
  var userXml: string
  var checksum: SciDACChecksum
  var hasRecordInfo = false
  var hasChecksum = false
  
  while r.reader.nextRecord() == lsSuccess:
    let ltype = r.reader.limeType
    
    if ltype == SciDACPrivateRecordXml:
      let xml = r.reader.readString()
      recordInfo = parseRecordXml(xml)
      hasRecordInfo = true
    elif ltype == SciDACRecordXml:
      userXml = r.reader.readString()
    elif ltype == SciDACChecksumRecord:
      let xml = r.reader.readString()
      checksum = parseChecksumXml(xml)
      r.storedChecksum = checksum
      r.hasChecksum = true
      hasChecksum = true
    elif ltype == SciDACBinaryData and hasRecordInfo:
      yield (recordInfo, userXml, checksum)
      hasRecordInfo = false
      hasChecksum = false
      userXml = ""
      checksum = initSciDACChecksum()

proc readBinaryData*(r: SciDACReader): seq[byte] =
  ## Read the binary data from current position
  ## Call this after iterating with `messages`
  ## Also reads the following checksum record if present
  result = r.reader.readAllData()
  
  # Try to read the next record to see if it's a checksum
  if r.reader.nextRecord() == lsSuccess:
    let ltype = r.reader.limeType
    if ltype == SciDACChecksumRecord:
      let xml = r.reader.readString()
      r.storedChecksum = parseChecksumXml(xml)
      r.hasChecksum = true
    # If not a checksum, we leave it for the next iteration
    # (but we've consumed a record - this is a limitation)

proc readBinaryDataTyped*[T](r: SciDACReader): seq[T] =
  ## Read binary data with type conversion
  let bytes = r.readBinaryData()
  let count = bytes.len div sizeof(T)
  result = newSeq[T](count)
  if count > 0:
    copyMem(addr result[0], unsafeAddr bytes[0], bytes.len)

# ===========================================================================
# High-level SciDAC writer
# ===========================================================================

type
  SciDACWriter* = ref object
    writer*: LimeWriter
    fileInfo*: SciDACFileInfo
    wroteFileInfo: bool

proc newSciDACWriter*(filename: string, fileInfo: SciDACFileInfo): SciDACWriter =
  ## Create a new SciDAC file for writing
  result = SciDACWriter(
    writer: newLimeWriter(filename),
    fileInfo: fileInfo,
    wroteFileInfo: false
  )

proc close*(w: SciDACWriter): LimeStatus =
  w.writer.close()

proc writeFileInfo*(w: SciDACWriter, userXml: string = ""): LimeStatus =
  ## Write the file-level metadata
  if w.wroteFileInfo:
    return lsSuccess
    
  let xml = generateFileXml(w.fileInfo)
  
  # File info is a single-record message if no user XML
  # Otherwise, two records in one message
  if userXml.len == 0:
    result = w.writer.writeRecord(SciDACPrivateFileXml, xml)
  else:
    result = w.writer.writeRecord(SciDACPrivateFileXml, xml, mbFlag=true, meFlag=false)
    if result == lsSuccess:
      result = w.writer.writeRecord(SciDACFileXml, userXml, mbFlag=false, meFlag=true)
  
  w.wroteFileInfo = true

proc writeField*[T](w: SciDACWriter, 
                    data: openArray[T],
                    recordInfo: SciDACRecordInfo,
                    userXml: string = ""): LimeStatus =
  ## Write a lattice field with metadata
  
  # Ensure file info is written first
  if not w.wroteFileInfo:
    result = w.writeFileInfo()
    if result != lsSuccess:
      return result
  
  let recordXml = generateRecordXml(recordInfo)
  let dataBytes = data.len * sizeof(T)
  
  # Write message: private-record-xml, user-record-xml (optional), binary-data
  result = w.writer.writeRecord(SciDACPrivateRecordXml, recordXml, 
                                 mbFlag=true, meFlag=false)
  if result != lsSuccess: return result
  
  if userXml.len > 0:
    result = w.writer.writeRecord(SciDACRecordXml, userXml,
                                   mbFlag=false, meFlag=false)
    if result != lsSuccess: return result
  
  # Write binary data
  let header = createHeader(false, true, SciDACBinaryData, dataBytes.int64)
  result = w.writer.writeHeader(header)
  if result != lsSuccess: return result
  
  if data.len > 0:
    let (_, status) = w.writer.writeData(unsafeAddr data[0], dataBytes)
    result = status

proc writeFieldBytes*(w: SciDACWriter,
                      data: openArray[byte],
                      recordInfo: SciDACRecordInfo,
                      userXml: string = ""): LimeStatus =
  ## Write raw bytes as field data
  w.writeField(data, recordInfo, userXml)

# ===========================================================================
# ILDG-specific utilities
# ===========================================================================

type
  ILDGReader* = ref object
    reader*: LimeReader
    formatInfo*: ILDGFormatInfo
    lfn*: string
    hasFormat: bool

proc newILDGReader*(filename: string): ILDGReader =
  ## Open an ILDG gauge configuration file
  result = ILDGReader(
    reader: newLimeReader(filename),
    hasFormat: false
  )

proc close*(r: ILDGReader) =
  r.reader.close()

proc readFormat*(r: ILDGReader): ILDGFormatInfo =
  ## Read ILDG format information
  while r.reader.nextRecord() == lsSuccess:
    if r.reader.limeType == ILDGFormat:
      let xml = r.reader.readString()
      r.formatInfo = parseILDGFormatXml(xml)
      r.hasFormat = true
      return r.formatInfo
    elif r.reader.limeType == ILDGDataLFN:
      r.lfn = r.reader.readString()
  
  raise newException(IOError, "No ILDG format info found")

proc readGaugeField*(r: ILDGReader): seq[byte] =
  ## Read the gauge field binary data
  while r.reader.nextRecord() == lsSuccess:
    if r.reader.limeType == ILDGBinaryData:
      return r.reader.readAllData()
  
  raise newException(IOError, "No ILDG binary data found")

# ===========================================================================
# Utility functions
# ===========================================================================

proc defaultRecordInfo*(precision: SciDACPrecision, 
                        colors: int = 3,
                        spins: int = 0,
                        typesize: int = 18,
                        datacount: int = 4): SciDACRecordInfo =
  ## Create default record info for gauge configurations
  let precStr = if precision == spSingle: "F" else: "D"
  SciDACRecordInfo(
    version: "1.0",
    date: now().format("ddd MMM dd HH:mm:ss yyyy"),
    recordtype: 0,
    datatype: fmt"QDP_{precStr}{colors}_ColorMatrix",
    precision: precision,
    colors: colors,
    spins: spins,
    typesize: typesize,
    datacount: datacount
  )

proc newFileInfo*(dims: openArray[int], volfmt: int = 0): SciDACFileInfo =
  ## Create file info from lattice dimensions
  SciDACFileInfo(
    version: "1.0",
    spacetime: dims.len,
    dims: @dims,
    volfmt: volfmt
  )

proc calcDataSize*(fileInfo: SciDACFileInfo, 
                   recordInfo: SciDACRecordInfo): int64 =
  ## Calculate expected binary data size
  var vol: int64 = 1
  for d in fileInfo.dims:
    vol *= d
  
  let floatSize = if recordInfo.precision == spSingle: 4 else: 8
  result = vol * recordInfo.typesize.int64 * floatSize.int64 * recordInfo.datacount.int64

when isMainModule:
  # Example usage
  block:
    echo "Creating SciDAC test file..."
    
    let dims = [4, 4, 4, 8]
    let fileInfo = newFileInfo(dims)
    let writer = newSciDACWriter("/tmp/test_scidac.lime", fileInfo)
    
    # Write file info
    discard writer.writeFileInfo("<userFileInfo>Test file</userFileInfo>")
    
    # Create test data (fake gauge field)
    var vol = 1
    for d in dims: vol *= d
    let numFloats = vol * 18 * 4  # 4 links per site, 18 reals per SU(3)
    var data = newSeq[float64](numFloats)
    for i in 0..<data.len:
      data[i] = float64(i) * 0.001
    
    let recordInfo = defaultRecordInfo(spDouble)
    discard writer.writeField(data, recordInfo, "<info>Test gauge field</info>")
    
    discard writer.close()
    echo "Done writing SciDAC file."
  
  block:
    echo "\nReading SciDAC test file..."
    echo "LIME contents:"
    dumpContents("/tmp/test_scidac.lime")
