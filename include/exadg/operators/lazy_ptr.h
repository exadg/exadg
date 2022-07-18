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

#ifndef LAZY_PTR_H_
#define LAZY_PTR_H_

namespace ExaDG
{
template<typename T>
class lazy_ptr
{
public:
  lazy_ptr() : t_ptr(&t)
  {
  }

  // resets the pointer (using own data)
  void
  reset()
  {
    this->t_ptr = &this->t;
  }

  // resets the pointer (using external data)
  void
  reset(T const & t_other)
  {
    this->t_ptr = &t_other;
  }

  // provides access to own data storage, e.g., in order to overwrite the data
  T &
  own()
  {
    return t;
  }

  T const * operator->() const
  {
    return t_ptr;
  }

  T const & operator*() const
  {
    return *t_ptr;
  }

private:
  T         t;
  T const * t_ptr;
};

} // namespace ExaDG

#endif
