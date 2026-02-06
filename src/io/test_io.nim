#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/test_io.nim
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

## Unit tests for LIME/SciDAC/QIO I/O facilities
##
## Run with:
##   cd local && make test_io && ./reliq -n 4 -e test_io

import std/[unittest, os, sequtils, strformat]
import io/io

const
  TestDir = "/tmp/reliq_io_tests"
  TestLimeFile = TestDir / "test.lime"
  TestSciDACFile = TestDir / "test_scidac.lime"
  TestQIOFile = TestDir / "test_qio.lime"

# ===========================================================================
# Test setup and teardown
# ===========================================================================

proc setupTestDir() =
  if not dirExists(TestDir):
    createDir(TestDir)

proc cleanupTestDir() =
  if dirExists(TestDir):
    removeDir(TestDir)

# ===========================================================================
# LIME format tests
# ===========================================================================

suite "LIME Format":
  setup:
    setupTestDir()
  
  teardown:
    discard  # Keep files for inspection; cleanup at end
  
  test "Write and read single record":
    let writer = newLimeWriter(TestLimeFile)
    let status = writer.writeRecord("test-type", "Hello, LIME!")
    check status == lsSuccess
    check writer.close() == lsSuccess
    
    let reader = newLimeReader(TestLimeFile)
    check reader.nextRecord() == lsSuccess
    check reader.limeType == "test-type"
    check reader.dataBytes == 12
    check reader.mbFlag == true
    check reader.meFlag == true
    
    let data = reader.readString()
    check data == "Hello, LIME!"
    reader.close()
  
  test "Write and read multiple records in one message":
    let writer = newLimeWriter(TestLimeFile)
    check writer.writeRecord("type-1", "First record", mbFlag=true, meFlag=false) == lsSuccess
    check writer.writeRecord("type-2", "Second record", mbFlag=false, meFlag=true) == lsSuccess
    check writer.close() == lsSuccess
    
    let reader = newLimeReader(TestLimeFile)
    
    check reader.nextRecord() == lsSuccess
    check reader.limeType == "type-1"
    check reader.mbFlag == true
    check reader.meFlag == false
    check reader.readString() == "First record"
    
    check reader.nextRecord() == lsSuccess
    check reader.limeType == "type-2"
    check reader.mbFlag == false
    check reader.meFlag == true
    check reader.readString() == "Second record"
    
    check reader.nextRecord() == lsEndOfFile
    reader.close()
  
  test "Write and read binary data":
    let binaryData = @[byte(0), 1, 2, 3, 255, 254, 253, 252]
    
    let writer = newLimeWriter(TestLimeFile)
    check writer.writeRecord("binary-type", binaryData) == lsSuccess
    check writer.close() == lsSuccess
    
    let reader = newLimeReader(TestLimeFile)
    check reader.nextRecord() == lsSuccess
    check reader.dataBytes == 8
    
    let readData = reader.readAllData()
    check readData == binaryData
    reader.close()
  
  test "Multiple messages in one file":
    let writer = newLimeWriter(TestLimeFile)
    # Message 1
    check writer.writeRecord("msg1-rec1", "Message 1, Record 1", mbFlag=true, meFlag=true) == lsSuccess
    # Message 2
    check writer.writeRecord("msg2-rec1", "Message 2, Record 1", mbFlag=true, meFlag=false) == lsSuccess
    check writer.writeRecord("msg2-rec2", "Message 2, Record 2", mbFlag=false, meFlag=true) == lsSuccess
    check writer.close() == lsSuccess
    
    let reader = newLimeReader(TestLimeFile)
    var msgCount = 0
    var recCount = 0
    
    for header in reader.records:
      inc recCount
      if header.mbFlag:
        inc msgCount
    
    check msgCount == 2
    check recCount == 3
    reader.close()
  
  test "Padding alignment":
    # LIME requires 8-byte alignment padding
    # Write data of various sizes and verify we can read back correctly
    for size in [1, 7, 8, 9, 15, 16, 17]:
      var data = newSeq[byte](size)
      for i in 0..<size:
        data[i] = byte(i mod 256)
      
      let writer = newLimeWriter(TestLimeFile)
      check writer.writeRecord("padding-test", data) == lsSuccess
      check writer.close() == lsSuccess
      
      let reader = newLimeReader(TestLimeFile)
      check reader.nextRecord() == lsSuccess
      check reader.dataBytes == size
      let readData = reader.readAllData()
      check readData == data
      reader.close()
  
  test "Iterator over records":
    let writer = newLimeWriter(TestLimeFile)
    for i in 1..5:
      check writer.writeRecord(fmt"record-{i}", fmt"Data {i}") == lsSuccess
    check writer.close() == lsSuccess
    
    let reader = newLimeReader(TestLimeFile)
    var count = 0
    for header in reader.records:
      inc count
      check header.limeType == fmt"record-{count}"
    check count == 5
    reader.close()

# ===========================================================================
# SciDAC format tests
# ===========================================================================

suite "SciDAC Format":
  setup:
    setupTestDir()
  
  test "Create and read file info":
    let dims = [4, 4, 4, 8]
    let fileInfo = newFileInfo(dims)
    
    check fileInfo.spacetime == 4
    check fileInfo.dims == @[4, 4, 4, 8]
    check fileInfo.volfmt == 0
  
  test "XML generation and parsing":
    let fileInfo = SciDACFileInfo(
      version: "1.0",
      spacetime: 4,
      dims: @[8, 8, 8, 16],
      volfmt: 0
    )
    
    let xml = generateFileXml(fileInfo)
    check xml.contains("<spacetime>4</spacetime>")
    check xml.contains("<dims>8 8 8 16</dims>")
    
    let parsed = parseFileXml(xml)
    check parsed.spacetime == 4
    check parsed.dims == @[8, 8, 8, 16]
  
  test "Write and read SciDAC file":
    let dims = [4, 4, 4, 8]
    let fileInfo = newFileInfo(dims)
    let writer = newSciDACWriter(TestSciDACFile, fileInfo)
    
    check writer.writeFileInfo("<userInfo>Test</userInfo>") == lsSuccess
    
    # Create test data
    var vol = 1
    for d in dims: vol *= d
    let numFloats = vol * 18 * 4  # 4 links, 18 reals per SU(3)
    var data = newSeq[float64](numFloats)
    for i in 0..<data.len:
      data[i] = float64(i) * 0.001
    
    let recordInfo = defaultRecordInfo(spDouble)
    check writer.writeField(data, recordInfo, "<field>gauge</field>") == lsSuccess
    check writer.close() == lsSuccess
    
    # Verify file exists and has correct structure
    check fileExists(TestSciDACFile)
    
    let reader = newLimeReader(TestSciDACFile)
    var foundFileXml = false
    var foundRecordXml = false
    var foundBinaryData = false
    
    for header in reader.records:
      if header.limeType == SciDACPrivateFileXml:
        foundFileXml = true
      elif header.limeType == SciDACPrivateRecordXml:
        foundRecordXml = true
      elif header.limeType == SciDACBinaryData:
        foundBinaryData = true
        check header.dataLength == int64(numFloats * sizeof(float64))
    
    check foundFileXml
    check foundRecordXml
    check foundBinaryData
    reader.close()
  
  test "Default record info":
    let recInfo = defaultRecordInfo(spDouble, colors=3, spins=0)
    check recInfo.precision == spDouble
    check recInfo.colors == 3
    check recInfo.spins == 0
    check recInfo.typesize == 18  # 3x3 complex = 18 reals
    check recInfo.datacount == 4

# ===========================================================================
# QIO format tests
# ===========================================================================

suite "QIO Format":
  setup:
    setupTestDir()
  
  test "Lexicographic index conversion":
    let dims = [4, 4, 4, 8]
    
    # Test a few known coordinates
    check lexIndex(@[0, 0, 0, 0], dims) == 0
    check lexIndex(@[1, 0, 0, 0], dims) == 1
    check lexIndex(@[0, 1, 0, 0], dims) == 4
    check lexIndex(@[0, 0, 1, 0], dims) == 16
    check lexIndex(@[0, 0, 0, 1], dims) == 64
    
    # Test round-trip for all sites
    var vol = 1
    for d in dims: vol *= d
    
    for idx in 0..<vol:
      let coords = lexCoords(idx, dims)
      let backIdx = lexIndex(coords, dims)
      check backIdx == idx
  
  test "Checksum calculation":
    var cs = initChecksum()
    let data = [byte(1), 2, 3, 4, 5, 6, 7, 8]
    cs.update(data, 0)
    
    # Checksum should be non-zero
    check cs.suma != 0 or cs.sumb != 0
    
    # Same data should give same checksum
    var cs2 = initChecksum()
    cs2.update(data, 0)
    check cs.suma == cs2.suma
    check cs.sumb == cs2.sumb
  
  test "Checksum combine":
    var cs1 = initChecksum()
    var cs2 = initChecksum()
    cs1.update([byte(1), 2, 3, 4], 0)
    cs2.update([byte(5), 6, 7, 8], 1)
    
    let combined = combine(cs1, cs2)
    # Combined checksum should be different from individuals
    check combined.suma != cs1.suma or combined.sumb != cs1.sumb
  
  test "Byte swapping 32-bit":
    var data = [byte(0x01), 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
    swapBytes32(data)
    check data == [byte(0x04), 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05]
    
    # Swap back
    swapBytes32(data)
    check data == [byte(0x01), 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
  
  test "Byte swapping 64-bit":
    var data = [byte(1), 2, 3, 4, 5, 6, 7, 8]
    swapBytes64(data)
    check data == [byte(8), 7, 6, 5, 4, 3, 2, 1]
    
    # Swap back
    swapBytes64(data)
    check data == [byte(1), 2, 3, 4, 5, 6, 7, 8]
  
  test "Write and read gauge field":
    let dims: array[4, int] = [4, 4, 4, 8]
    let vol = volume(dims)
    let numFloats = vol * 4 * 18  # 4 links * 18 reals per SU(3)
    
    var data = newSeq[float64](numFloats)
    for i in 0..<data.len:
      data[i] = float64(i) * 0.0001
    
    let field = GaugeField[float64](dims: dims, data: data)
    
    check writeGaugeField(TestQIOFile, field, "<info>Test config</info>") == lsSuccess
    check fileExists(TestQIOFile)
    
    # Read back and verify
    let readField = readGaugeField(TestQIOFile)
    check readField.dims == dims
    check readField.data.len == data.len
    
    # Check data values (allowing for potential byte order conversion)
    for i in 0..<min(100, data.len):
      check abs(readField.data[i] - data[i]) < 1e-10
  
  test "Volume calculation":
    check volume([4, 4, 4, 8]) == 512
    check volume([8, 8, 8, 16]) == 8192
    check volume([16, 16, 16, 32]) == 262144

# ===========================================================================
# Integration tests
# ===========================================================================

suite "Integration":
  setup:
    setupTestDir()
  
  test "Full write-read cycle with verification":
    # Write a complete SciDAC gauge configuration
    let dims = [4, 4, 4, 8]
    var vol = 1
    for d in dims: vol *= d
    
    # Create reproducible test data
    let numFloats = vol * 4 * 18
    var originalData = newSeq[float64](numFloats)
    for i in 0..<originalData.len:
      originalData[i] = sin(float64(i) * 0.01) * cos(float64(i) * 0.007)
    
    let field = GaugeField[float64](dims: dims, data: originalData)
    let userXml = """<info>
  <beta>6.0</beta>
  <trajectory>1000</trajectory>
  <algorithm>HMC</algorithm>
</info>"""
    
    check writeGaugeField(TestQIOFile, field, userXml) == lsSuccess
    
    # Read back
    let readField = readGaugeField(TestQIOFile)
    
    # Verify dimensions
    check readField.dims == dims
    
    # Verify data length
    check readField.data.len == originalData.len
    
    # Verify data values
    var maxDiff = 0.0
    for i in 0..<originalData.len:
      let diff = abs(readField.data[i] - originalData[i])
      if diff > maxDiff:
        maxDiff = diff
    
    check maxDiff < 1e-14  # Should be exact for IEEE float64
  
  test "dumpContents utility":
    # Create a file and use dumpContents (just verify it doesn't crash)
    let writer = newLimeWriter(TestLimeFile)
    discard writer.writeRecord("test", "Test data")
    discard writer.close()
    
    # This should print to stdout without crashing
    dumpContents(TestLimeFile)

# ===========================================================================
# Cleanup
# ===========================================================================

suite "Cleanup":
  test "Remove test directory":
    cleanupTestDir()
    check not dirExists(TestDir)
