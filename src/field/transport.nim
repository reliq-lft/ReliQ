import reliq

import scalarfield
import tensorfield

import lattice/[simplecubiclattice]
import kokkos/[kokkosdispatch]

type TransporterKind* = enum tkShift, tkCovariantShift

type Transporter*[D: static[int], T] = ref object of RootObj
  ## Transporter type for shifting fields on a lattice
  ##
  ## Represents a transporter that can shift fields along a specified
  ## direction and distance on a given lattice.
  case kind*: TransporterKind
    of tkShift: discard
    of tkCovariantShift:
      transporter*: TensorField[D, T]
  lattice*: SimpleCubicLattice[D]
  direction*: int
  distance*: int

type Transporters*[D: static[int], T] = array[D, Transporter[D, T]]

#[ Transporter constructors ]#

proc defaultDistances[D: static[int]](): array[D, int] =
  for i in 0..<D: result[i] = 1

proc newTransporter*[D: static[int]](
  lattice: SimpleCubicLattice[D],
  direction: int,
  distance: int = 1
): Transporter[D, float] =
  ## Create a new Transporter for shifting fields
  ##
  ## Parameters:
  ## - `lattice`: The lattice on which the transporter operates
  ## - `direction`: The direction index for the shift
  ## - `distance`: The distance to shift along the specified direction
  ## 
  ## Returns:
  ## A new Transporter instance configured for non-covariant shifts
  ## 
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let transporter = newTransporter(lattice, 0, 1)
  ## ```
  return Transporter[D, float](
    kind: tkShift,
    lattice: lattice,
    direction: direction,
    distance: distance
  )

proc newTranspoter*[D: static[int], T](
  transporter: Tensor[D, T],
  direction: int,
  distance: int = 1
): Transporter[D, T] =
  ## Create a new Transporter for covariant shifts
  ##
  ## Parameters:
  ## - `transporter`: The tensor field representing the transporter
  ## - `direction`: The direction index for the shift
  ## - `distance`: The distance to shift along the specified direction
  ## 
  ## Returns:
  ## A new Transporter instance configured for covariant shifts
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let transporterField = lattice.newTensor(@[4, 4]): float
  ## let transporter = newTranspoter(transporterField, 0, 1)
  ## ```
  result = Transporter[D, T](
    kind: tkCovariantShift,
    transporter: transpoter,
    lattice: transporter.lattice,
    direction: direction,
    distance: distance
  )

proc newTransporters*[D: static[int], T](
  lattice: SimpleCubicLattice[D],
  distances: array[D, int] = defaultDistances[D]()
): Transporters[D, T] =
  ## Create an array of Transporters for all directions
  ##
  ## Parameters:
  ## - `lattice`: The lattice on which the transporters operate
  ## - `distances`: An array specifying the distance to shift for each direction
  ##
  ## Returns:
  ## An array of Transporter instances for each direction
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let distances = [1, 0, -1, 0]
  ## let transporters = newTransporters(lattice, distances)
  ## ```
  for dir in 0..<D:
    result[dir] = newTransporter[D, T](lattice, dir, distances[dir])
    # do halo exchange if grid is padded

proc newTransporters*[D: static[int], T](
  transporters: GaugeTensor[D, T],
  distances: array[D, int] = defaultDistances[D]()
): Transporters[D, T] =
  ## Create an array of Transporters for all directions using covariant transporters
  ##
  ## Parameters:
  ## - `transporters`: An array of tensor fields representing the transporters for each direction
  ## - `distances`: An array specifying the distance to shift for each direction
  ##
  ## Returns:
  ## An array of Transporter instances for each direction
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## var transporterFields: array[4, Tensor[4, float]]
  ## for dir in 0..<4:
  ##   transporterFields[dir] = lattice.newTensor(@[4, 4]): float
  ## let distances = [1, 0, -1, 0]
  ## let transporters = newTransporters(transporterFields, distances)
  ## ```
  for dir in 0..<D:
    result[dir] = newTranspoter[D, T](transporters[dir], dir, distances[dir])
    # do halo exchange if grid is padded

#[ Transporter operations ]#

template transport*[D: static[int], T](
  transporter: Transporter[D, T],
  field: Field[D, T]
): Field[D, T] =
  ## Transport a field using the specified transporter
  ##
  ## Parameters:
  ## - `transporter`: The Transporter instance to use for shifting
  ## - `field`: The field to be transported
  ##
  ## Returns:
  ## A new Field instance that has been shifted according to the transporter
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let field = newField(lattice): float
  ## let transporter = newTransporter(lattice, 0, 1)
  ## let transportedField = transport(transporter, field)
  ## ```
  # do halo exchange if grid is padded
  field

template `*`*[D: static[int], T](
  transporter: Transporter[D, T],
  field: Field[D, T]
): Field[D, T] =
  ## Operator overload for transporting a field using the specified transporter
  ##
  ## Parameters:
  ## - `transporter`: The Transporter instance to use for shifting
  ## - `field`: The field to be transported
  ##
  ## Returns:
  ## A new Field instance that has been shifted according to the transporter
  ##
  ## Example:
  ## ```nim
  ## let lattice = newSimpleCubicLattice([8, 8, 8, 16])
  ## let field = newField(lattice): float
  ## let transporter = newTransporter(lattice, 0, 1)
  ## let transportedField = transporter * field
  ## ```
  transport(transporter, field)