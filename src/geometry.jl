abstract type AbstractGeometry end

struct Star1 <: AbstractGeometry end
struct Star2 <: AbstractGeometry end
struct Star3 <: AbstractGeometry end

struct Square1 <: AbstractGeometry end
struct Square2 <: AbstractGeometry end
struct Square3 <: AbstractGeometry end

const Star = Union{Star1, Star2, Star3}
const Square = Union{Square1, Square2, Star3}