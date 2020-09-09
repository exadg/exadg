/*
 * integrator_flags.h
 *
 *  Created on: Jun 14, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_OPERATORS_INTEGRATOR_FLAGS_H_
#define INCLUDE_EXADG_OPERATORS_INTEGRATOR_FLAGS_H_

namespace ExaDG
{
using namespace dealii;

struct CellFlags
{
  CellFlags(const bool value = false, const bool gradient = false, const bool hessian = false)
    : value(value), gradient(gradient), hessian(hessian){};

  CellFlags
  operator||(CellFlags const & other)
  {
    CellFlags cell_flags_combined;

    cell_flags_combined.value    = this->value || other.value;
    cell_flags_combined.gradient = this->gradient || other.gradient;
    cell_flags_combined.hessian  = this->hessian || other.hessian;

    return cell_flags_combined;
  }

  bool value;
  bool gradient;
  bool hessian;
};

struct FaceFlags
{
  FaceFlags(const bool value = false, const bool gradient = false)
    : value(value), gradient(gradient){};

  FaceFlags
  operator||(FaceFlags const & other)
  {
    FaceFlags face_flags_combined;

    face_flags_combined.value    = this->value || other.value;
    face_flags_combined.gradient = this->gradient || other.gradient;

    return face_flags_combined;
  }

  bool
  do_eval() const
  {
    return value || gradient;
  }

  bool value;
  bool gradient;
};

struct IntegratorFlags
{
  IntegratorFlags()
  {
  }

  IntegratorFlags
  operator||(IntegratorFlags const & other)
  {
    IntegratorFlags flags_combined;

    flags_combined.cell_evaluate  = this->cell_evaluate || other.cell_evaluate;
    flags_combined.cell_integrate = this->cell_integrate || other.cell_integrate;

    flags_combined.face_evaluate  = this->face_evaluate || other.face_evaluate;
    flags_combined.face_integrate = this->face_integrate || other.face_integrate;

    return flags_combined;
  }

  CellFlags cell_evaluate;
  CellFlags cell_integrate;

  FaceFlags face_evaluate;
  FaceFlags face_integrate;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_INTEGRATOR_FLAGS_H_ */
