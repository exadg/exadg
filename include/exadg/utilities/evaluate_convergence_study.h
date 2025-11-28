/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

#ifndef EXADG_UTILITIES_EVALUATE_CONVERGENCE_STUDY_H_
#define EXADG_UTILITIES_EVALUATE_CONVERGENCE_STUDY_H_

namespace ExaDG
{
/*
 * This function searches for files in the application directories created following a
 * `run_<run_id>_<fieldname>_<error_type>` logic. It assumes refining in space, time or polynomial
 * degree with increasing `run_id`.
 */
void
evaluate_convergence_study(MPI_Comm const & mpi_comm, std::string const & input_parameter_file);

} // namespace ExaDG

#endif /* EXADG_UTILITIES_EVALUATE_CONVERGENCE_STUDY_H_ */
