# I/O Module

ReliQ provides a unified I/O system for reading and writing lattice field
data in standard lattice QCD formats.  The system supports distributed
parallel I/O via Global Arrays — rank 0 reads/writes the file and broadcasts
data to all ranks.

## Supported Formats

| Format | Read | Write | Description |
|--------|------|-------|-------------|
| LIME | ✓ | ✓ | Low-level binary container format |
| SciDAC | ✓ | ✓ | XML metadata + binary data with checksums |
| ILDG | ✓ | ✓ | International Lattice Data Grid gauge configs |
| QIO | ✓ | ✓ | SciDAC QIO parallel I/O format |

## Quick Start

### Reading a Gauge Configuration

```nim
import reliq

parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    # Create 4 gauge link fields (one per direction)
    var g0 = lat.newTensorField([3, 3]): Complex64
    var g1 = lat.newTensorField([3, 3]): Complex64
    var g2 = lat.newTensorField([3, 3]): Complex64
    var g3 = lat.newTensorField([3, 3]): Complex64
    var gaugeField = [g0, g1, g2, g3]

    # Read ILDG gauge configuration
    readGaugeField(gaugeField, "config.ildg")
```

### Writing a Tensor Field

```nim
parallel:
  let lat = newSimpleCubicLattice([8, 8, 8, 16])

  block:
    var field = lat.newTensorField([3, 3]): float64

    # Initialize field...
    var local = field.newLocalTensorField()
    for n in all 0..<local.numSites():
      var site = local.getSite(n)
      site[0, 0] = 1.0
    local.releaseLocalTensorField()

    # Write to LIME/SciDAC format
    writeTensorField(field, "output.lime")
```

## LIME Container Format

LIME (Lattice Interchange Message Encapsulation) is a binary container format
that stores typed records:

```nim
import reliq

# Write LIME records
var writer = newLimeWriter("output.lime")
writer.writeRecord("text/xml", xmlData)
writer.writeRecord("application/binary", binaryData)
writer.close()

# Read LIME records
var reader = newLimeReader("output.lime")
for record in reader.records():
  echo record.typeName
  echo record.dataSize
reader.close()
```

### Record Structure

Each LIME record has:
- A type string (e.g., ``"text/xml"``, ``"application/binary"``)
- A data payload
- Automatic 8-byte alignment padding

Multiple records can be grouped into LIME messages.

## SciDAC Format

SciDAC files wrap binary lattice data with XML metadata and checksums:

```nim
import reliq

# Reader with metadata access
var reader = newTensorFieldReader("data.scidac")
echo reader.fileInfo     # XML file metadata
echo reader.recordInfo   # XML record metadata
echo reader.precision    # "F" for float32, "D" for float64
echo reader.colors       # Number of color indices
echo reader.spins        # Number of spin indices
echo reader.dims         # Lattice dimensions

# Checksum validation
echo reader.hasChecksum
echo reader.storedChecksum
echo reader.computedChecksum
```

### XML Metadata

SciDAC files contain structured XML with:
- **File info**: creation date, archive, description
- **Record info**: precision, colors, spins, typesize, datacount
- **Checksum**: CRC32 for data validation

## TensorField I/O API

### Reading

```nim
# Read a single tensor field
readTensorField(field, "data.scidac")

# Read with byte-swapping (for endianness conversion)
readTensorField(field, "data.scidac", swapBytes=true)

# Read a gauge field (array of D tensor fields)
readGaugeField(gaugeField, "config.ildg")
```

### Writing

```nim
# Write a single tensor field
writeTensorField(field, "output.lime")

# Write with custom XML metadata
writeTensorField(field, "output.lime", userXml="<custom>data</custom>")

# Write a gauge field
writeGaugeField(gaugeField, "config.ildg")
```

### Coordinate Mapping

The I/O system handles the mapping between file-order (lexicographic) and
GA-order (distributed) coordinates:

```nim
# Global lexicographic index for a coordinate
let idx = globalLexIndex(coords, lattDims)

# Convert lexicographic index to coordinates
let coords = globalLexCoords(idx, lattDims)
```

## Data Flow: Read

1. Rank 0 reads the binary data from the LIME container
2. Data is byte-swapped if needed (endianness detection)
3. Rank 0 creates a ``LocalTensorField`` and fills it with file data
4. ``releaseLocalTensorField()`` copies data back to the distributed GA
5. ``GA_Sync()`` ensures all ranks see the updated data

## Data Flow: Write

1. ``newLocalTensorField()`` copies rank-local GA data to contiguous buffer
2. Each rank's local data is gathered to rank 0
3. Rank 0 writes the LIME container with SciDAC/ILDG headers
4. Checksums are computed and stored

## Checksums

The QIO module provides checksum computation compatible with the SciDAC
standard:

- **CRC32** checksums for data validation
- Checksums are combined across MPI ranks
- Stored checksums are compared against computed checksums on read

## Module Reference

| Module | Description |
|--------|-------------|
| [io/io](io/io.html) | Top-level I/O module, re-exports all sub-modules |
| [io/tensorio](io/tensorio.html) | ``readTensorField``, ``writeTensorField``, ``readGaugeField`` |
