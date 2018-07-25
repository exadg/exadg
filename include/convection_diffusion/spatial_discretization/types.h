#ifndef CONVECTION_DIFFUSION_TYPES
#define CONVECTION_DIFFUSION_TYPES

namespace ConvDiff
{

enum class OperatorType {
  full,
  homogeneous,
  inhomogeneous
};

enum class BoundaryType {
  undefined,
  dirichlet,
  neumann
};

}

#endif