#ifndef CONVECTION_DIFFUSION_TYPES
#define CONVECTION_DIFFUSION_TYPES

#include "../../operators/operator_type.h"

namespace ConvDiff
{
enum class BoundaryType
{
  undefined,
  dirichlet,
  neumann
};

} // namespace ConvDiff

#endif