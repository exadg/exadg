/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
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
