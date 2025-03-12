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

#ifndef INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_LEVELS_HYBRID_MULTIGRID_H_
#define INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_LEVELS_HYBRID_MULTIGRID_H_

namespace ExaDG
{
struct MGDoFHandlerIdentifier
{
  MGDoFHandlerIdentifier(unsigned int degree, bool is_dg) : degree(degree), is_dg(is_dg)
  {
  }

  bool
  operator<(MGDoFHandlerIdentifier const & other) const
  {
    return not((degree >= other.degree) and (is_dg >= other.is_dg));
  }

  bool
  operator==(MGDoFHandlerIdentifier const & other) const
  {
    return (degree == other.degree) and (is_dg == other.is_dg);
  }

  unsigned int degree;
  bool         is_dg;
};

struct MGLevelInfo
{
  MGLevelInfo(unsigned int h_level, unsigned int dealii_tria_level, unsigned int degree, bool is_dg)
    : _h_level(h_level), _dealii_tria_level(dealii_tria_level), _dof_handler_id(degree, is_dg)
  {
  }

  MGLevelInfo(unsigned int           h_level,
              unsigned int           dealii_tria_level,
              MGDoFHandlerIdentifier dof_handler_id)
    : _h_level(h_level), _dealii_tria_level(dealii_tria_level), _dof_handler_id(dof_handler_id)
  {
  }

  unsigned int
  h_level() const
  {
    return _h_level;
  }

  unsigned int
  dealii_tria_level() const
  {
    return _dealii_tria_level;
  }

  unsigned int
  degree() const
  {
    return _dof_handler_id.degree;
  }

  bool
  is_dg() const
  {
    return _dof_handler_id.is_dg;
  }

  MGDoFHandlerIdentifier
  dof_handler_id() const
  {
    return _dof_handler_id;
  }

private:
  /**
   * Counter for h-level running from 0 (coarse) to n_h_levels (fine)
   */
  unsigned int _h_level;

  /**
   * Triangulation level in dealii nomenclature, which is -1 in case we have a separate
   * Triangulation object for each multigrid h-level (MG h-level = active level of Triangulation
   * object corresponding to coarser MG h-level).
   */
  unsigned int _dealii_tria_level;


  MGDoFHandlerIdentifier _dof_handler_id;
};

} // namespace ExaDG


#endif /* INCLUDE_EXADG_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_LEVELS_HYBRID_MULTIGRID_H_ */
