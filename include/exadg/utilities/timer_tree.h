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

#ifndef INCLUDE_EXADG_UTILITIES_TIMER_TREE_H_
#define INCLUDE_EXADG_UTILITIES_TIMER_TREE_H_

// C++
#include <memory>
#include <string>
#include <vector>

// deal.II
#include <deal.II/base/conditional_ostream.h>

namespace ExaDG
{
class TimerTree
{
public:
  /**
   * Constructor. Initializes ID with an empty string.
   */
  TimerTree();

  /**
   * This function clears the content of this tree. Sub trees inserted
   * into this tree via pointers to external trees are not touched.
   */
  void
  clear();

  /**
   * This function inserts a measured wall_time into the tree, by
   * either creating a new entry in the tree if this function is called
   * the first time with this ID, or by adding the wall_time to an
   * entry already existing in the tree.
   */
  void
  insert(std::vector<std::string> const ids, double const wall_time);

  /**
   * This function inserts a whole sub_tree into an existing tree, where
   * the parameter names specifies the place at which to insert the sub_tree.
   * This function allows to combine different timer trees in a modular way.
   * If a non empty string new_name is provided, the id of  sub_tree is
   * replaced by new_name when inserted into the tree.
   */
  void
  insert(std::vector<std::string>   ids,
         std::shared_ptr<TimerTree> sub_tree,
         std::string const          new_name = "");

  /**
   * Prints wall time of all items of a tree without an analysis of
   * the relative share of the children.
   */
  void
  print_plain(dealii::ConditionalOStream const & pcout) const;

  /**
   * This is the actual function of interest of this class, i.e., an
   * analysis of wall times with a hierarchical formatting of results.
   * Relative wall times are printed for all children of a sub-tree if
   * a wall time has been set for the parent of these children. In this
   * case, an additional item `other` is created in order to give insights
   * to which extent the code has been covered with timers and to which
   * extend time is spent is other code paths that are currently not
   * covered by timers.
   */
  void
  print_level(dealii::ConditionalOStream const & pcout, unsigned int const level) const;

  /**
   * Returns the maximum number of levels of the timer tree.
   */
  unsigned int
  get_max_level() const;

private:
  /**
   * This function "copies" a tree, meaning that only the ID is copied, while
   * pointers to data and to sub-trees still point to the original tree "other".
   */
  void
  copy_from(std::shared_ptr<TimerTree> other);

  /**
   * This function erases the first entry of a vector of strings.
   */
  std::vector<std::string>
  erase_first(std::vector<std::string> const & in) const;

  /**
   * This function computes and returns the MPI-average wall time for the
   * underlying data object.
   */
  double
  get_average_wall_time() const;

  /**
   * This function returns the number of characters needed by the "longest"
   * item of the tree, in order to ensure a nice formatting when printing the tree.
   */
  unsigned int
  get_length() const;

  /**
   * This function prints the whole tree by recursively going through all sub-trees.
   * An offset acting as an indentation is applied from one level to the next in order
   * to obtain a nice formatting when printing the tree.
   */
  void
  do_print_plain(dealii::ConditionalOStream const & pcout,
                 unsigned int const                 offset,
                 unsigned int const                 length) const;

  /**
   * This function prints the whole tree up to a specified level.
   */
  void
  do_print_level(dealii::ConditionalOStream const & pcout,
                 unsigned int const                 level,
                 unsigned int const                 offset,
                 unsigned int const                 length) const;

  /**
   * This function print the name ID of the root element of the present tree. The
   * boolean parameter new_line describes whether a line-break is applied after
   * printing the name.
   */
  void
  print_name(dealii::ConditionalOStream const & pcout,
             unsigned int const                 offset,
             unsigned int const                 length,
             bool const                         new_line) const;

  /**
   * This function prints the root element of the current tree. The name ID is
   * printed always, while data is only printed if it is available. The boolean
   * parameter relative allows to also print the relative percentage in wall-time
   * compared to a refernce wall-time ref_time.
   */
  void
  print_own(dealii::ConditionalOStream const & pcout,
            unsigned int const                 offset,
            unsigned int const                 length,
            bool const                         relative = false,
            double const                       ref_time = -1.0) const;

  /**
   * This function prints the direct children of the root element of the current
   * tree. Depending on the sub-trees and the data available for the sub-trees,
   * the implementation decides whether relative timings can be printed and which
   * sub-trees are printed in this case.
   */
  void
  print_direct_children(dealii::ConditionalOStream const & pcout,
                        unsigned int const                 offset,
                        unsigned int const                 length,
                        bool const                         relative = false,
                        double const                       ref_time = -1.0) const;

  std::string id;

  struct Data
  {
    Data() : wall_time(0.0)
    {
    }

    double wall_time;
  };

  std::shared_ptr<Data> data;

  std::vector<std::shared_ptr<TimerTree>> sub_trees;

  static unsigned int const offset_per_level = 2;
  static unsigned int const precision        = 2;
};
} // namespace ExaDG


#endif /* INCLUDE_EXADG_UTILITIES_TIMER_TREE_H_ */
