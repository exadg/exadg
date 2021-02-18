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

#ifndef MG_TRANSFER_MF
#define MG_TRANSFER_MF

namespace ExaDG
{
template<typename VectorType>
class MGTransfer
{
public:
  virtual ~MGTransfer()
  {
  }

  virtual void
  interpolate(unsigned int const level, VectorType & dst, VectorType const & src) const = 0;

  virtual void
  restrict_and_add(unsigned int const level, VectorType & dst, VectorType const & src) const = 0;

  virtual void
  prolongate(unsigned int const level, VectorType & dst, VectorType const & src) const = 0;
};
} // namespace ExaDG

#endif
