#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/lime.nim
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

## LIME (Lattice QCD Interchange Message Encapsulation) format implementation
##
## This module provides reading and writing capabilities for LIME files,
## which are used by the SciDAC community for lattice QCD data interchange.
##
## LIME format specification:
## - Magic number: 0x456789AB (big-endian)
## - Header: 144 bytes (18 x 64-bit words)
## - Data payload with 8-byte alignment padding
##
## Reference: https://usqcd-software.github.io/c-lime/

import std/[streams, strutils, endians, sequtils]

const
  LimeMagicNumber*: uint32 = 0x456789AB'u32
  LimeVersion*: uint16 = 1
  LimeHeaderSize* = 144
  LimeTypeMaxLen* = 128
  LimeAlignment = 8

type
  LimeStatus* = enum
    lsSuccess = 0
    lsErrParam
    lsErrHeaderNext
    lsErrWrite
    lsEndOfRecord
    lsEndOfFile
    lsErrRead
    lsErrSeek
    lsErrMbMe
    lsErrClose
    lsErrMagic
    lsErrVersion

  LimeRecordHeader* = object
    mbFlag*: bool       ## Message Begin flag
    meFlag*: bool       ## Message End flag  
    limeType*: string   ## LIME type identifier (max 128 chars)
    dataLength*: int64  ## Length of data payload in bytes

  LimeReader* = ref object
    stream: Stream
    currentHeader: LimeRecordHeader
    headerRead: bool
    bytesRemaining: int64
    recordPosition: int64

  LimeWriter* = ref object
    stream: Stream
    currentHeader: LimeRecordHeader
    headerWritten: bool
    bytesWritten: int64
    needsClosing: bool

# ===========================================================================
# Byte order utilities (LIME uses big-endian)
# ===========================================================================

proc toBigEndian32(x: uint32): uint32 {.inline.} =
  when cpuEndian == littleEndian:
    var tmp = x
    var res: uint32
    bigEndian32(addr res, addr tmp)
    result = res
  else:
    result = x

proc fromBigEndian32(x: uint32): uint32 {.inline.} =
  toBigEndian32(x)

proc toBigEndian16(x: uint16): uint16 {.inline.} =
  when cpuEndian == littleEndian:
    var tmp = x
    var res: uint16
    bigEndian16(addr res, addr tmp)
    result = res
  else:
    result = x

proc fromBigEndian16(x: uint16): uint16 {.inline.} =
  toBigEndian16(x)

proc toBigEndian64(x: int64): int64 {.inline.} =
  when cpuEndian == littleEndian:
    var tmp = x
    var res: int64
    bigEndian64(addr res, addr tmp)
    result = res
  else:
    result = x

proc fromBigEndian64(x: int64): int64 {.inline.} =
  toBigEndian64(x)

# ===========================================================================
# LimeReader implementation
# ===========================================================================

proc newLimeReader*(stream: Stream): LimeReader =
  ## Create a new LIME reader from an existing stream
  result = LimeReader(
    stream: stream,
    headerRead: false,
    bytesRemaining: 0,
    recordPosition: 0
  )

proc newLimeReader*(filename: string): LimeReader =
  ## Create a new LIME reader from a file
  let stream = newFileStream(filename, fmRead)
  if stream.isNil:
    raise newException(IOError, "Cannot open file for reading: " & filename)
  result = newLimeReader(stream)

proc close*(reader: LimeReader) =
  ## Close the LIME reader
  if not reader.stream.isNil:
    reader.stream.close()

proc calcPadding(dataLen: int64): int =
  ## Calculate padding bytes needed for 8-byte alignment
  let rem = dataLen mod LimeAlignment
  if rem == 0: 0 else: (LimeAlignment - rem).int

proc nextRecord*(reader: LimeReader): LimeStatus =
  ## Advance to the next LIME record and read its header
  ## Returns lsSuccess on success, lsEndOfFile at end of file
  
  # Skip remaining data and padding from current record
  if reader.headerRead and reader.bytesRemaining > 0:
    let padding = calcPadding(reader.currentHeader.dataLength)
    let skipBytes = reader.bytesRemaining + padding
    try:
      reader.stream.setPosition(reader.stream.getPosition() + skipBytes.int)
    except:
      return lsErrSeek
  elif reader.headerRead:
    # Skip just the padding
    let padding = calcPadding(reader.currentHeader.dataLength)
    if padding > 0:
      try:
        reader.stream.setPosition(reader.stream.getPosition() + padding)
      except:
        return lsErrSeek

  reader.recordPosition = reader.stream.getPosition()

  # Read 144-byte header
  var headerBuf: array[LimeHeaderSize, byte]
  let bytesRead = reader.stream.readData(addr headerBuf[0], LimeHeaderSize)
  
  if bytesRead == 0:
    return lsEndOfFile
  if bytesRead < LimeHeaderSize:
    return lsErrRead

  # Parse word 0: magic number (32 bits), version (16 bits), flags (16 bits)
  var word0: uint64
  copyMem(addr word0, addr headerBuf[0], 8)
  
  # Extract components (big-endian)
  let magicBytes = cast[ptr uint32](addr headerBuf[0])[]
  let magic = fromBigEndian32(magicBytes)
  
  if magic != LimeMagicNumber:
    return lsErrMagic
  
  let versionBytes = cast[ptr uint16](addr headerBuf[4])[]
  let version = fromBigEndian16(versionBytes)
  
  # Flags are in bytes 6-7
  let flags = headerBuf[6]
  let mbFlag = (flags and 0x80) != 0  # Bit 48 = MB
  let meFlag = (flags and 0x40) != 0  # Bit 49 = ME
  
  # Parse word 1: data length (64-bit big-endian)
  var dataLenBE: int64
  copyMem(addr dataLenBE, addr headerBuf[8], 8)
  let dataLength = fromBigEndian64(dataLenBE)
  
  # Parse LIME type (128 bytes, null-terminated ASCII)
  var limeType = newString(LimeTypeMaxLen)
  copyMem(addr limeType[0], addr headerBuf[16], LimeTypeMaxLen)
  # Find null terminator
  let nullPos = limeType.find('\0')
  if nullPos >= 0:
    limeType.setLen(nullPos)
  limeType = limeType.strip()
  
  reader.currentHeader = LimeRecordHeader(
    mbFlag: mbFlag,
    meFlag: meFlag,
    limeType: limeType,
    dataLength: dataLength
  )
  reader.headerRead = true
  reader.bytesRemaining = dataLength
  
  return lsSuccess

proc header*(reader: LimeReader): LimeRecordHeader =
  ## Get the current record header
  reader.currentHeader

proc mbFlag*(reader: LimeReader): bool =
  ## Check if current record is message begin
  reader.currentHeader.mbFlag

proc meFlag*(reader: LimeReader): bool =
  ## Check if current record is message end
  reader.currentHeader.meFlag

proc limeType*(reader: LimeReader): string =
  ## Get the LIME type string of current record
  reader.currentHeader.limeType

proc dataBytes*(reader: LimeReader): int64 =
  ## Get the total data bytes in current record
  reader.currentHeader.dataLength

proc bytesRemaining*(reader: LimeReader): int64 =
  ## Get remaining unread bytes in current record
  reader.bytesRemaining

proc readData*(reader: LimeReader, dest: pointer, nbytes: int): (int, LimeStatus) =
  ## Read data from the current record
  ## Returns (bytes_read, status)
  if not reader.headerRead:
    return (0, lsErrParam)
  
  let toRead = min(nbytes.int64, reader.bytesRemaining).int
  if toRead == 0:
    return (0, lsEndOfRecord)
  
  let bytesRead = reader.stream.readData(dest, toRead)
  reader.bytesRemaining -= bytesRead
  
  if bytesRead < toRead:
    return (bytesRead, lsErrRead)
  
  if reader.bytesRemaining == 0:
    return (bytesRead, lsEndOfRecord)
  
  return (bytesRead, lsSuccess)

proc readAllData*(reader: LimeReader): seq[byte] =
  ## Read all remaining data from current record
  if not reader.headerRead or reader.bytesRemaining == 0:
    return @[]
  
  result = newSeq[byte](reader.bytesRemaining)
  let (bytesRead, _) = reader.readData(addr result[0], result.len)
  result.setLen(bytesRead)

proc readString*(reader: LimeReader): string =
  ## Read record data as a string (for ASCII/XML records)
  let data = reader.readAllData()
  result = newString(data.len)
  if data.len > 0:
    copyMem(addr result[0], unsafeAddr data[0], data.len)
  # Strip null terminators
  let nullPos = result.find('\0')
  if nullPos >= 0:
    result.setLen(nullPos)

proc seek*(reader: LimeReader, offset: int64, whence: int = 0): LimeStatus =
  ## Seek within the current record payload
  ## whence: 0 = SEEK_SET (from start), 1 = SEEK_CUR (from current)
  if not reader.headerRead:
    return lsErrParam
  
  let dataStart = reader.recordPosition + LimeHeaderSize
  let dataEnd = dataStart + reader.currentHeader.dataLength
  
  var newPos: int64
  case whence
  of 0:  # SEEK_SET
    newPos = dataStart + offset
  of 1:  # SEEK_CUR
    newPos = reader.stream.getPosition() + offset
  else:
    return lsErrParam
  
  # Clamp to valid range
  if newPos < dataStart:
    newPos = dataStart
  if newPos > dataEnd:
    newPos = dataEnd
  
  try:
    reader.stream.setPosition(newPos.int)
    reader.bytesRemaining = dataEnd - newPos
    return lsSuccess
  except:
    return lsErrSeek

proc getPosition*(reader: LimeReader): int64 =
  ## Get current file position
  reader.stream.getPosition()

proc setPosition*(reader: LimeReader, offset: int64): LimeStatus =
  ## Set absolute file position (must point to start of a record)
  try:
    reader.stream.setPosition(offset.int)
    reader.headerRead = false
    reader.bytesRemaining = 0
    return lsSuccess
  except:
    return lsErrSeek

# ===========================================================================
# LimeWriter implementation
# ===========================================================================

proc newLimeWriter*(stream: Stream): LimeWriter =
  ## Create a new LIME writer from an existing stream
  result = LimeWriter(
    stream: stream,
    headerWritten: false,
    bytesWritten: 0,
    needsClosing: false
  )

proc newLimeWriter*(filename: string): LimeWriter =
  ## Create a new LIME writer to a file
  let stream = newFileStream(filename, fmWrite)
  if stream.isNil:
    raise newException(IOError, "Cannot open file for writing: " & filename)
  result = newLimeWriter(stream)

proc writeHeader*(writer: LimeWriter, header: LimeRecordHeader): LimeStatus =
  ## Write a LIME record header
  if writer.headerWritten and writer.bytesWritten < writer.currentHeader.dataLength:
    return lsErrHeaderNext
  
  # Write any pending padding from previous record
  if writer.headerWritten:
    let padding = calcPadding(writer.currentHeader.dataLength)
    if padding > 0:
      var zeros: array[8, byte]
      writer.stream.writeData(addr zeros[0], padding)
  
  var headerBuf: array[LimeHeaderSize, byte]
  
  # Word 0: magic (32 bits) + version (16 bits) + flags (16 bits)
  let magicBE = toBigEndian32(LimeMagicNumber)
  copyMem(addr headerBuf[0], unsafeAddr magicBE, 4)
  
  let versionBE = toBigEndian16(LimeVersion)
  copyMem(addr headerBuf[4], unsafeAddr versionBE, 2)
  
  var flags: byte = 0
  if header.mbFlag: flags = flags or 0x80
  if header.meFlag: flags = flags or 0x40
  headerBuf[6] = flags
  headerBuf[7] = 0  # Reserved
  
  # Word 1: data length
  let dataLenBE = toBigEndian64(header.dataLength)
  copyMem(addr headerBuf[8], unsafeAddr dataLenBE, 8)
  
  # Words 2-17: LIME type (128 bytes)
  let typeLen = min(header.limeType.len, LimeTypeMaxLen)
  if typeLen > 0:
    copyMem(addr headerBuf[16], unsafeAddr header.limeType[0], typeLen)
  
  writer.stream.writeData(addr headerBuf[0], LimeHeaderSize)
  writer.currentHeader = header
  writer.headerWritten = true
  writer.bytesWritten = 0
  writer.needsClosing = not header.meFlag
  
  return lsSuccess

proc createHeader*(mbFlag, meFlag: bool, limeType: string, dataLen: int64): LimeRecordHeader =
  ## Create a LIME record header
  LimeRecordHeader(
    mbFlag: mbFlag,
    meFlag: meFlag,
    limeType: limeType,
    dataLength: dataLen
  )

proc writeData*(writer: LimeWriter, source: pointer, nbytes: int): (int, LimeStatus) =
  ## Write data to the current record
  if not writer.headerWritten:
    return (0, lsErrParam)
  
  let remaining = writer.currentHeader.dataLength - writer.bytesWritten
  let toWrite = min(nbytes.int64, remaining).int
  
  if toWrite == 0:
    return (0, lsEndOfRecord)
  
  writer.stream.writeData(source, toWrite)
  writer.bytesWritten += toWrite
  
  return (toWrite, lsSuccess)

proc writeString*(writer: LimeWriter, s: string): LimeStatus =
  ## Write a string as record data
  if s.len > 0:
    let (_, status) = writer.writeData(unsafeAddr s[0], s.len)
    return status
  return lsSuccess

proc writeBytes*(writer: LimeWriter, data: openArray[byte]): LimeStatus =
  ## Write bytes as record data
  if data.len > 0:
    let (_, status) = writer.writeData(unsafeAddr data[0], data.len)
    return status
  return lsSuccess

proc close*(writer: LimeWriter): LimeStatus =
  ## Close the LIME writer, finalizing the file
  # Write final padding if needed
  if writer.headerWritten:
    let padding = calcPadding(writer.currentHeader.dataLength)
    if padding > 0:
      var zeros: array[8, byte]
      writer.stream.writeData(addr zeros[0], padding)
  
  if not writer.stream.isNil:
    writer.stream.close()
  
  return lsSuccess

# ===========================================================================
# Convenience utilities
# ===========================================================================

iterator records*(reader: LimeReader): LimeRecordHeader =
  ## Iterate over all records in a LIME file
  while reader.nextRecord() == lsSuccess:
    yield reader.header

proc writeRecord*(writer: LimeWriter, limeType: string, data: string,
                  mbFlag = true, meFlag = true): LimeStatus =
  ## Write a complete record with string data
  let header = createHeader(mbFlag, meFlag, limeType, data.len.int64)
  result = writer.writeHeader(header)
  if result == lsSuccess:
    result = writer.writeString(data)

proc writeRecord*(writer: LimeWriter, limeType: string, data: openArray[byte],
                  mbFlag = true, meFlag = true): LimeStatus =
  ## Write a complete record with binary data
  let header = createHeader(mbFlag, meFlag, limeType, data.len.int64)
  result = writer.writeHeader(header)
  if result == lsSuccess:
    result = writer.writeBytes(data)

proc dumpContents*(filename: string) =
  ## Print contents of a LIME file (similar to lime_contents utility)
  let reader = newLimeReader(filename)
  defer: reader.close()
  
  var msgNum = 0
  var recNum = 0
  
  for header in reader.records:
    if header.mbFlag:
      inc msgNum
      recNum = 0
    inc recNum
    
    echo "Message: ", msgNum, " Record: ", recNum
    echo "  Type: ", header.limeType
    echo "  Bytes: ", header.dataLength
    echo "  MB: ", header.mbFlag, " ME: ", header.meFlag
    
    # Print ASCII content for small text records
    if header.dataLength < 4096 and header.dataLength > 0:
      let data = reader.readString()
      if data.len > 0 and data.allIt(it.ord >= 32 and it.ord < 127 or it in {'\n', '\r', '\t'}):
        echo "  Content:"
        for line in data.splitLines:
          echo "    ", line
    echo ""

when isMainModule:
  import std/sequtils
  
  # Test: Create a simple LIME file
  block:
    echo "Creating test LIME file..."
    let writer = newLimeWriter("/tmp/test.lime")
    
    # Message 1: Two records
    discard writer.writeRecord("test-type-1", "Hello, LIME!", mbFlag=true, meFlag=false)
    discard writer.writeRecord("test-type-2", "Second record in message", mbFlag=false, meFlag=true)
    
    # Message 2: Single record
    discard writer.writeRecord("binary-data", @[byte(1), 2, 3, 4, 5, 6, 7, 8])
    
    discard writer.close()
    echo "Done writing."
  
  # Test: Read the file back
  block:
    echo "\nReading test LIME file..."
    dumpContents("/tmp/test.lime")
