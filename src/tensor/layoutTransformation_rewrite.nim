## Proposed rewrite of layoutTransformation from tensorview.nim
##
## Two bugs in the original:
##
##   Bug 1 — Missing innerBlockSize multiplier on the site address.
##   paddedHostIdx is a lex count of sites (stride 1), but src has
##   inter-site stride = product(paddedShape).  The correct base is
##   coords.coordsToLex(paddedGrid) * innerBlockSize.
##
##   Bug 2 — "+ tensorPadding" on elemCoords double-counts the inner
##   ghost offset.  accessPadded() already shifts the raw pointer past
##   innerPaddedOffset, so src[siteBase + 0] == Re(0,0).  Adding
##   tensorPadding again offsets into the ghost fringe.  The padded
##   strides encoded in paddedShape already describe the gapped layout;
##   just address with the unpadded element coords:
##     elemIdx.lexToCoords(shape).coordsToLex(paddedShape)
##
## Ghost zones:
##   The simdGrid is fixed by the local grid.  The padded device grid
##   inherits that same decomposition: paddedDeviceGrid = paddedGrid div simdGrid.
##   Iterating over paddedDeviceGrid instead of deviceGrid covers local +
##   ghost sites in the same loop with no special-casing.  Padded host coords
##   are reconstructed as:
##     paddedHostCoords = paddedDeviceCoords * simdGrid + laneCoords
##   which replaces deviceIdxAndLaneIdxToHostIdx (that function only mapped
##   into the local lex space, making ghost sites unreachable).

proc layoutTransformation*[D: static[int], R: static[int], S](
  src: LocalStorage[S];
  tensorShape: array[R, int];
  latticePadding: array[D, int];
  simdLayout: SIMDLayout[D];
  T: typedesc;
): LocalStorage[S] =
  let grid = simdLayout.hostGrid
  let simdGrid = simdLayout.simdGrid
  let paddedGrid = grid + 2 * latticePadding
  let paddedDeviceGrid = paddedGrid div simdGrid

  var shape: array[R+1, int]
  for r in 0..<R: shape[r] = tensorShape[r]
  shape[R] = (if isComplex(T): 2 else: 1)
  let paddedShape = shape + 2   # innerGhostWidth = 1, broadcast over all R+1 dims

  let innerBlockSize = product(paddedShape)
  let numSIMDSites = simdLayout.numSIMDSites
  let numPaddedDeviceSites = product(paddedDeviceGrid)
  let elementsPerSite = product(shape)
  let reshape = [numPaddedDeviceSites, elementsPerSite, numSIMDSites]

  var target = cast[LocalStorage[S]](alloc(product(reshape) * sizeof(S)))

  threads:
    for paddedDeviceIdx in 0..<numPaddedDeviceSites:
      for laneIdx in 0..<numSIMDSites:
        let paddedHostBase =
          (paddedDeviceIdx.lexToCoords(paddedDeviceGrid) * simdGrid +
           laneIdx.lexToCoords(simdGrid)).coordsToLex(paddedGrid) * innerBlockSize
        for elemIdx in 0..<elementsPerSite:
          let targetIdx = [paddedDeviceIdx, elemIdx, laneIdx].coordsToLex(reshape)
          target[targetIdx] = src[paddedHostBase + elemIdx.lexToCoords(shape).coordsToLex(paddedShape)]

  return target
