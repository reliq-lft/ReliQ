#[ 
  ReliQ lattice field theory framework: https://github.com/reliq-lft/ReliQ
  Source file: src/io/io.nim
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

## ReliQ I/O Module
##
## This module provides unified I/O facilities for lattice field data,
## supporting multiple file formats used in the lattice QCD community:
##
## - **LIME**: Low-level container format (Lattice QCD Interchange Message Encapsulation)
## - **SciDAC**: Standard format with XML metadata built on LIME
## - **ILDG**: International Lattice Data Grid format for gauge configurations
## - **QIO**: Full parallel I/O compatible with USQCD software stack
##
## Example usage:
## ```nim
## import io
##
## # Read a gauge configuration
## let gauge = readGaugeField("config.lime")
## echo "Lattice dimensions: ", gauge.dims
##
## # Write a gauge configuration  
## discard writeGaugeField("output.lime", gauge, "<info>My config</info>")
##
## # Low-level LIME access
## let reader = newLimeReader("file.lime")
## for header in reader.records:
##   echo "Record type: ", header.limeType
## reader.close()
## ```
##
## References:
## - LIME: https://usqcd-software.github.io/c-lime/
## - QIO: https://github.com/usqcd-software/qio

import lime
import scidac
import qio
import tensorio

export lime
export scidac
export qio
export tensorio

when isMainModule:
  import std/[unittest, os, strformat, math, strutils]
  
  import ../parallel
  import ../lattice
  import ../tensor/[globaltensor, localtensor]
  import ../globalarrays/[gawrap]
  import ../utils/[complex]
  from ../lattice/simplecubiclattice import SimpleCubicLattice
  
  parallel:
    # Get MPI rank using GA after initialization
    let mpiRank = GA_Nodeid()
    let TestDir = "/tmp/reliq_io_tests_" & $mpiRank
    let SharedTestDir = "/tmp/reliq_io_tests_shared"  # Shared across all ranks
    let TestLimeFile = TestDir / "test.lime"
    let TestSciDACFile = TestDir / "test_scidac.lime"
    let TestQIOFile = TestDir / "test_qio.lime"
    let TestTensorFile = SharedTestDir / "test_tensor.lime"

    # =========================================================================
    # Test setup and teardown
    # =========================================================================

    proc setupTestDir() =
      if not dirExists(TestDir):
        createDir(TestDir)
      # Only rank 0 creates shared directory
      if mpiRank == 0 and not dirExists(SharedTestDir):
        createDir(SharedTestDir)
      GA_Sync()

    proc cleanupTestDir() =
      if dirExists(TestDir):
        removeDir(TestDir)
      # Only rank 0 cleans shared directory
      GA_Sync()
      if mpiRank == 0 and dirExists(SharedTestDir):
        removeDir(SharedTestDir)
      GA_Sync()

    # =========================================================================
    # LIME format tests
    # =========================================================================

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
        check writer.writeRecord("msg1-rec1", "Message 1, Record 1", mbFlag=true, meFlag=true) == lsSuccess
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

    # =========================================================================
    # SciDAC format tests
    # =========================================================================

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
        
        var vol = 1
        for d in dims: vol *= d
        let numFloats = vol * 18 * 4
        var data = newSeq[float64](numFloats)
        for i in 0..<data.len:
          data[i] = float64(i) * 0.001
        
        let recordInfo = defaultRecordInfo(spDouble)
        check writer.writeField(data, recordInfo, "<field>gauge</field>") == lsSuccess
        check writer.close() == lsSuccess
        
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
        check recInfo.typesize == 18
        check recInfo.datacount == 4

    # =========================================================================
    # QIO format tests
    # =========================================================================

    suite "QIO Format":
      setup:
        setupTestDir()
      
      test "Lexicographic index conversion":
        let dims = [4, 4, 4, 8]
        
        check lexIndex(@[0, 0, 0, 0], dims) == 0
        check lexIndex(@[1, 0, 0, 0], dims) == 1
        check lexIndex(@[0, 1, 0, 0], dims) == 4
        check lexIndex(@[0, 0, 1, 0], dims) == 16
        check lexIndex(@[0, 0, 0, 1], dims) == 64
        
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
        
        check cs.suma != 0 or cs.sumb != 0
        
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
        check combined.suma != cs1.suma or combined.sumb != cs1.sumb
      
      test "Byte swapping 32-bit":
        var data = [byte(0x01), 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        swapBytes32(data)
        check data == [byte(0x04), 0x03, 0x02, 0x01, 0x08, 0x07, 0x06, 0x05]
        
        swapBytes32(data)
        check data == [byte(0x01), 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
      
      test "Byte swapping 64-bit":
        var data = [byte(1), 2, 3, 4, 5, 6, 7, 8]
        swapBytes64(data)
        check data == [byte(8), 7, 6, 5, 4, 3, 2, 1]
        
        swapBytes64(data)
        check data == [byte(1), 2, 3, 4, 5, 6, 7, 8]
      
      test "Write and read gauge field":
        let dims: array[4, int] = [4, 4, 4, 8]
        let vol = volume(dims)
        let numFloats = vol * 4 * 18
        
        var data = newSeq[float64](numFloats)
        for i in 0..<data.len:
          data[i] = float64(i) * 0.0001
        
        let field = GaugeField[float64](dims: dims, data: data)
        
        check writeGaugeField(TestQIOFile, field, "<info>Test config</info>") == lsSuccess
        check fileExists(TestQIOFile)
        
        let readField = readGaugeField(TestQIOFile)
        check readField.dims == dims
        check readField.data.len == data.len
        
        for i in 0..<min(100, data.len):
          check abs(readField.data[i] - data[i]) < 1e-10
      
      test "Volume calculation":
        check volume([4, 4, 4, 8]) == 512
        check volume([8, 8, 8, 16]) == 8192
        check volume([16, 16, 16, 32]) == 131072

    # =========================================================================
    # Integration tests
    # =========================================================================

    suite "Integration":
      setup:
        setupTestDir()
      
      test "Full write-read cycle with verification":
        let dims = [4, 4, 4, 8]
        var vol = 1
        for d in dims: vol *= d
        
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
        
        let readField = readGaugeField(TestQIOFile)
        
        check readField.dims == dims
        check readField.data.len == originalData.len
        
        var maxDiff = 0.0
        for i in 0..<originalData.len:
          let diff = abs(readField.data[i] - originalData[i])
          if diff > maxDiff:
            maxDiff = diff
        
        check maxDiff < 1e-14
      
      test "dumpContents utility":
        let writer = newLimeWriter(TestLimeFile)
        discard writer.writeRecord("test", "Test data")
        discard writer.close()
        
        dumpContents(TestLimeFile)

    # =========================================================================
    # TensorField I/O tests
    # =========================================================================

    const SampleILDGFile = "src/io/sample/ildg.lat"

    suite "TensorField I/O":
      setup:
        setupTestDir()
        GA_Sync()  # Ensure all ranks are synchronized
      
      test "Read sample ILDG gauge config":
        # The sample file has dims 8 8 8 16 and contains gauge field data
        # Gauge field: 4 directions, each a [3,3] complex matrix field
        let dims: array[4, int] = [8, 8, 8, 16]
        let lattice = newSimpleCubicLattice(dims)
        
        # Create 4 tensor fields for the 4 link directions
        var gaugeField0 = lattice.newTensorField([3, 3]): Complex64
        var gaugeField1 = lattice.newTensorField([3, 3]): Complex64
        var gaugeField2 = lattice.newTensorField([3, 3]): Complex64
        var gaugeField3 = lattice.newTensorField([3, 3]): Complex64
        var gaugeField = [gaugeField0, gaugeField1, gaugeField2, gaugeField3]
        
        if GA_Nodeid() == 0:
          echo "Reading sample ILDG file: ", SampleILDGFile
          
        readGaugeField(gaugeField, SampleILDGFile)
        
        # Verify we read something non-zero
        var nonZeroCount = 0
        for mu in 0..<4:
          var localField = gaugeField[mu].newLocalTensorField()
          let numElems = min(25, localField.numElements())
          for i in 0..<numElems:
            if abs(localField.data[i]) > 1e-15:
              inc nonZeroCount
        
        if GA_Nodeid() == 0:
          echo "  Non-zero elements (first 100 across all dirs): ", nonZeroCount
        
        check nonZeroCount > 0
      
      test "Write and read real TensorField":
        let dims: array[4, int] = [4, 4, 4, 8]
        let lattice = newSimpleCubicLattice(dims)
        
        var tensorA = lattice.newTensorField([3, 3]): float64
        var localA = tensorA.newLocalTensorField()
        
        let numSites = localA.numSites()
        for site in 0..<numSites:
          let base = site * 9
          for i in 0..<9:
            localA.data[base + i] = float64(site * 10 + i) * 0.001
        
        localA.releaseLocalTensorField()
        
        let status = writeTensorField(tensorA, TestTensorFile, "<info>Test real tensor</info>")
        check status == lsSuccess
        
        if mpiRank == 0:
          check fileExists(TestTensorFile)
        
        var tensorB = lattice.newTensorField([3, 3]): float64
        
        readTensorField(tensorB, TestTensorFile)
        
        var localB = tensorB.newLocalTensorField()
        for site in 0..<min(10, numSites):
          let base = site * 9
          for i in 0..<9:
            let expected = float64(site * 10 + i) * 0.001
            let diff = abs(localB.data[base + i] - expected)
            check diff < 1e-10
      
      test "Write and read complex TensorField":
        let dims: array[4, int] = [4, 4, 4, 8]
        let lattice = newSimpleCubicLattice(dims)
        
        var tensorA = lattice.newTensorField([3, 3]): Complex64
        var localA = tensorA.newLocalTensorField()
        
        let numSites = localA.numSites()
        for site in 0..<numSites:
          let base = site * 9 * 2
          for i in 0..<9:
            localA.data[base + i * 2] = float64(site * 10 + i) * 0.001
            localA.data[base + i * 2 + 1] = float64(site * 10 + i) * 0.0001
        
        localA.releaseLocalTensorField()
        
        let status = writeTensorField(tensorA, TestTensorFile, "<info>Test complex tensor</info>")
        check status == lsSuccess
        
        if mpiRank == 0:
          check fileExists(TestTensorFile)
        
        var tensorB = lattice.newTensorField([3, 3]): Complex64
        
        readTensorField(tensorB, TestTensorFile)
        
        var localB = tensorB.newLocalTensorField()
        for site in 0..<min(10, numSites):
          let base = site * 9 * 2
          for i in 0..<9:
            let expectedRe = float64(site * 10 + i) * 0.001
            let expectedIm = float64(site * 10 + i) * 0.0001
            let diffRe = abs(localB.data[base + i * 2] - expectedRe)
            let diffIm = abs(localB.data[base + i * 2 + 1] - expectedIm)
            check diffRe < 1e-10
            check diffIm < 1e-10
      
      test "TensorFieldReader metadata":
        let dims: array[4, int] = [4, 4, 4, 8]
        let lattice = newSimpleCubicLattice(dims)
        var tensorA = lattice.newTensorField([3, 3]): float64
        discard writeTensorField(tensorA, TestTensorFile)
        
        # Only rank 0 checks metadata (file reading is single-rank operation)
        if mpiRank == 0:
          let reader = newTensorFieldReader(TestTensorFile)
          defer: reader.close()
          
          check reader.dims.len == 4
          check reader.dims[0] == 4
          check reader.dims[1] == 4
          check reader.dims[2] == 4
          check reader.dims[3] == 8
      
      test "Lexicographic index conversions":
        let dims: array[4, int] = [4, 4, 4, 8]
        
        for idx in [0, 1, 15, 63, 255, 511]:
          let coords = globalLexCoords[4](idx, dims)
          let backIdx = globalLexIndex(coords, dims)
          check backIdx == idx
        
        check globalLexIndex([0, 0, 0, 0], dims) == 0
        check globalLexIndex([1, 0, 0, 0], dims) == 1
        check globalLexIndex([0, 1, 0, 0], dims) == 4
        check globalLexIndex([0, 0, 1, 0], dims) == 16
        check globalLexIndex([0, 0, 0, 1], dims) == 64

    # =========================================================================
    # Cleanup
    # =========================================================================

    suite "Cleanup":
      test "Remove test directory":
        cleanupTestDir()
        check not dirExists(TestDir)
