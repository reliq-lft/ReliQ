## Diagnostic: Check GA memory layout and siteOffsets

import std/[strformat, endians, math]

import lattice
import parallel
import tensor/[tensor]
import utils/[complex]
import gauge/gaugefield
import io/[tensorio, gaugeio]
import globalarrays/[gawrap, gatypes]

const SampleFile = "src/io/sample/ildg.lat"

when isMainModule:
  parallel:
    let dims: array[4, int] = [8, 8, 8, 16]
    let lat = newSimpleCubicLattice(dims)
    let ctx = newGaugeFieldContext[Complex64](gkSU3, rkFundamental)
    var u = lat.newGaugeField(ctx)
    gaugeio.readGaugeField(u, SampleFile, validate = false)

    local:
      const D = 4
      const R = 2
      const rank = D + R + 1  # 7
      let handle = u.field[0].data.getHandle()
      var lo, hi: array[rank, cint]
      var ld: array[rank-1, cint]
      var p: pointer
      let pid = GA_Nodeid()

      NGA_Distribution(handle, pid, addr lo[0], addr hi[0])
      NGA_Access(handle, addr lo[0], addr hi[0], addr p, addr ld[0])

      echo "=== GA Memory Layout ==="
      echo "lo: ", lo
      echo "hi: ", hi
      echo "ld (leading dims): ", ld
      echo ""

      # What localSiteOffset computes:
      let innerPadded = innerPaddedBlockSize(R, 1)
      echo "innerPaddedBlockSize(R=2, gw=1) = ", innerPadded  # Should be 27

      # What the actual GA stride should be:
      # For [8,8,8,16,3,3,2] with ghost [0,0,0,0,1,1,1]
      # Padded inner: [3+2, 3+2, 2+2] = [5, 5, 4]
      # Actual inner block = 5*5*4 = 100
      let actualInnerPadded = (3 + 2) * (3 + 2) * (2 + 2)
      echo "Actual padded inner block (5*5*4) = ", actualInnerPadded

      # Compute what the actual inner block should be from ld
      # ld gives leading dimensions: ld[0] is the stride to go 1 step in dim 1
      # For a 7D array in C order, ld[i] should be prod of padded sizes from dim (i+1) to (ndim-1)
      # But GA uses Fortran order (first dim fastest)
      echo ""
      echo "ld[0] = ", ld[0], " (stride for next element in dim 1, i.e. y)"
      echo "Expected if innerPadded=27: ", 8 * 27
      echo "Expected if innerPadded=100: ", 8 * 100

      # Direct memory probe: read first element at sites 0 and 1
      let rawPtr = cast[ptr UncheckedArray[float64]](p)
      echo ""
      echo "=== Direct GA memory probe ==="
      echo "p[0..1] (site 0, elem 0 re,im): ", rawPtr[0], ", ", rawPtr[1]

      # If stride is innerPadded=27, site 1 starts at 27
      echo "p[27..28] (site 1 if stride=27): ", rawPtr[27], ", ", rawPtr[28]
      # If stride is actualInnerPadded=100, site 1 starts at 100  
      echo "p[100..101] (site 1 if stride=100): ", rawPtr[100], ", ", rawPtr[101]

      # Now check what the raw binary says site 0 and site 1 should be
      let reader = newTensorFieldReader(SampleFile)
      reader.loadBinaryData(validate = false)
      let rawData = reader.binaryData
      reader.close()

      let nc = 3
      let bytesPerDir = nc * nc * 16
      let bytesPerSite = 4 * bytesPerDir

      # Raw site 0, mu=0, element (0,0)
      var re0, im0: float64
      copyMem(addr re0, unsafeAddr rawData[0], 8)
      copyMem(addr im0, unsafeAddr rawData[8], 8)
      var re0be, im0be: float64
      swapEndian64(addr re0be, addr re0)
      swapEndian64(addr im0be, addr im0)
      echo ""
      echo "Raw binary site 0, mu=0, (0,0) = (", re0be, ", ", im0be, ")"

      # Raw site 1, mu=0, element (0,0)
      var re1, im1: float64
      copyMem(addr re1, unsafeAddr rawData[bytesPerSite + 0], 8)
      copyMem(addr im1, unsafeAddr rawData[bytesPerSite + 8], 8)
      var re1be, im1be: float64
      swapEndian64(addr re1be, addr re1)
      swapEndian64(addr im1be, addr im1)
      echo "Raw binary site 1, mu=0, (0,0) = (", re1be, ", ", im1be, ")"

      # Now check siteOffsets
      var lf = u.field[0].newLocalTensorField()
      echo ""
      echo "siteOffsets[0] = ", lf.siteOffsets[0]
      echo "siteOffsets[1] = ", lf.siteOffsets[1]
      echo "siteOffsets[2] = ", lf.siteOffsets[2]
      echo "siteOffsets[8] = ", lf.siteOffsets[8]

      # Check if data at siteOffset[1] matches raw site 1
      let off1 = lf.siteOffsets[1]
      echo ""
      echo "GA data at siteOffset[1]=", off1, ": re=", rawPtr[off1], " im=", rawPtr[off1+1]
      echo "Matches raw site 1? ", abs(rawPtr[off1] - re1be) < 1e-14

      # Try probing at various strides to find where site 1 actually is
      echo ""
      echo "=== Searching for site 1 data in GA memory ==="
      for stride in 1..200:
        if abs(rawPtr[stride] - re1be) < 1e-14 and abs(rawPtr[stride+1] - im1be) < 1e-14:
          echo fmt"  Found site 1 at GA offset {stride}"

      echo ""
      echo "=== Plaquette ==="
      let stencilPlaq = u.plaquette()
      echo fmt"Stencil plaquette:   {stencilPlaq:.15g}"
      echo  "Reference plaquette: 0.951249133097204"
