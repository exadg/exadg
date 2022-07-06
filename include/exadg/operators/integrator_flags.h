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

#include <deal.II/matrix_free/evaluation_flags.h>

namespace ExaDG
{
struct IntegratorFlags
{
  IntegratorFlags
  operator|(IntegratorFlags const & other)
  {
    IntegratorFlags flags_combined;

    flags_combined.cell_evaluate  = this->cell_evaluate | other.cell_evaluate;
    flags_combined.cell_integrate = this->cell_integrate | other.cell_integrate;

    flags_combined.face_evaluate  = this->face_evaluate | other.face_evaluate;
    flags_combined.face_integrate = this->face_integrate | other.face_integrate;

    return flags_combined;
  }

  dealii::EvaluationFlags::EvaluationFlags cell_evaluate{dealii::EvaluationFlags::nothing};
  dealii::EvaluationFlags::EvaluationFlags cell_integrate{dealii::EvaluationFlags::nothing};

  dealii::EvaluationFlags::EvaluationFlags face_evaluate{dealii::EvaluationFlags::nothing};
  dealii::EvaluationFlags::EvaluationFlags face_integrate{dealii::EvaluationFlags::nothing};
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_INTEGRATOR_FLAGS_H_ */
