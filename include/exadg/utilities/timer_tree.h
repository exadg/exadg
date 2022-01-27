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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/exceptions.h>

namespace ExaDG
{
class TimerTree
{
public:
  TimerTree() : id("")
  {
  }

  /*
   * This function clears the content of this tree. Sub trees inserted
   * into this tree via pointers to external trees are not touched.
   */
  void
  clear()
  {
    this->id = "";
    data     = nullptr;
    sub_trees.clear();
  }

  /*
   * This function inserts a measured wall_time into the tree, by
   * either creating a new entry in the tree if this function is called
   * the first time with this ID, or by adding the wall_time to an
   * entry already existing in the tree.
   */
  void
  insert(std::vector<std::string> const ids, double const wall_time)
  {
    AssertThrow(ids.size() > 0, dealii::ExcMessage("empty name."));

    if(this->id == "") // the tree is currently empty
    {
      AssertThrow(sub_trees.empty(), dealii::ExcMessage("invalid state found. aborting."));

      this->id = ids[0];

      if(ids.size() == 1) // leaves of tree reached, insert the data
      {
        data = std::make_shared<Data>();
        data->wall_time += wall_time;

        return;
      }
      else // go deeper
      {
        std::vector<std::string> remaining_id = erase_first(ids);

        std::shared_ptr<TimerTree> new_tree = std::make_shared<TimerTree>();
        new_tree->insert(remaining_id, wall_time);
        sub_trees.push_back(new_tree);
      }
    }
    else if(this->id == ids[0]) // the tree already has some entries
    {
      if(ids.size() == 1) // leaves of tree reached, insert the data
      {
        if(data.get() == nullptr)
          data = std::make_shared<Data>();

        data->wall_time += wall_time;

        return;
      }
      else // find correct sub-tree or insert new sub-tree
      {
        std::vector<std::string> remaining_id = erase_first(ids);

        bool found = false;
        for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
        {
          // find out where to insert item
          if((*it)->id == remaining_id[0])
          {
            found = true;

            (*it)->insert(remaining_id, wall_time);
          }
        }

        if(found == false)
        {
          std::shared_ptr<TimerTree> new_tree = std::make_shared<TimerTree>();
          new_tree->insert(remaining_id, wall_time);
          sub_trees.push_back(new_tree);
        }
      }
    }
    else // the provided name does not fit to this tree
    {
      AssertThrow(false,
                  dealii::ExcMessage("The name provided is " + ids[0] + ", but must be " + id +
                                     " instead."));
    }
  }

  /*
   * This function inserts a whole sub_tree into an existing tree, where
   * the parameter names specifies the place at which to insert the sub_tree.
   * This function allows to combine different timer trees in a modular way.
   * If a non empty string new_name is provided, the id of  sub_tree is
   * replaced by new_name when inserted into the tree.
   */
  void
  insert(std::vector<std::string>   ids,
         std::shared_ptr<TimerTree> sub_tree,
         std::string const          new_name = "")
  {
    AssertThrow(ids.size() > 0, dealii::ExcMessage("Empty ID specified."));
    AssertThrow(id == ids[0], dealii::ExcMessage("Invalid ID specified."));

    std::vector<std::string> remaining_id = erase_first(ids);

    bool found = false;
    if(remaining_id.size() > 0)
    {
      for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
      {
        if((*it)->id == remaining_id[0])
        {
          (*it)->insert(remaining_id, sub_tree, new_name);
          found = true;
        }
      }
    }

    if(found == false)
    {
      AssertThrow(
        remaining_id.size() == 0,
        dealii::ExcMessage(
          "Subtree can not be inserted since the specified ID does not exist in this tree."));

      std::shared_ptr<TimerTree> new_tree(new TimerTree());
      new_tree->copy_from(sub_tree);
      if(!new_name.empty())
        new_tree->id = new_name;

      // Make sure that the new tree does not already exist
      for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
      {
        AssertThrow(
          new_tree->id != (*it)->id,
          dealii::ExcMessage(
            "Subtree can not be inserted since the tree already contains a subtree with the same ID."));
      }

      sub_trees.push_back(new_tree);
    }
  }

  /*
   * Prints wall time of all items of a tree without an analysis of
   * the relative share of the children.
   */
  void
  print_plain(dealii::ConditionalOStream const & pcout) const
  {
    unsigned int const length = get_length();

    pcout << std::endl;

    do_print_plain(pcout, 0, length);
  }

  /*
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
  print_level(dealii::ConditionalOStream const & pcout, unsigned int const level) const
  {
    unsigned int const length = get_length();

    pcout << std::endl;

    do_print_level(pcout, level, 0, length);
  }

private:
  void
  copy_from(std::shared_ptr<TimerTree> other)
  {
    *this = *other;
  }

  /*
   * This function erases the first entry of the vector.
   */
  std::vector<std::string>
  erase_first(std::vector<std::string> const & in) const
  {
    AssertThrow(in.size() > 0, dealii::ExcMessage("empty name."));

    std::vector<std::string> out(in);
    out.erase(out.begin());

    return out;
  }

  double
  get_average_wall_time() const
  {
    dealii::Utilities::MPI::MinMaxAvg time_data =
      dealii::Utilities::MPI::min_max_avg(data->wall_time, MPI_COMM_WORLD);
    return time_data.avg;
  }

  /*
   * print functions
   */
  unsigned int
  get_length() const
  {
    unsigned int length = id.length();

    for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
    {
      length = std::max(length, (*it)->get_length() + offset_per_level);
    }

    return length;
  }

  void
  do_print_plain(dealii::ConditionalOStream const & pcout,
                 unsigned int const                 offset,
                 unsigned int const                 length) const
  {
    if(id.empty())
      return;

    print_own(pcout, offset, length);

    for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
    {
      (*it)->do_print_plain(pcout, offset + offset_per_level, length);
    }
  }

  void
  do_print_level(dealii::ConditionalOStream const & pcout,
                 unsigned int const                 level,
                 unsigned int const                 offset,
                 unsigned int const                 length) const
  {
    if(id.empty())
      return;

    if(level == 0)
    {
      if(data.get())
        print_own(pcout, offset, length);
    }
    else if(level == 1)
    {
      if(sub_trees.size() > 0)
      {
        print_own(pcout, offset, length, true, data->wall_time);

        bool const relative = (data.get() != nullptr);
        print_direct_children(pcout, offset + offset_per_level, length, relative, data->wall_time);
      }
    }
    else
    {
      // only print name
      print_name(pcout, offset, length);

      // recursively print sub trees (decreasing the level and incrementing
      // the offset)
      for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
      {
        (*it)->do_print_level(pcout, level - 1, offset + offset_per_level, length);
      }
    }
  }

  void
  print_name(dealii::ConditionalOStream const & pcout,
             unsigned int const                 offset,
             unsigned int const                 length) const
  {
    pcout << std::setw(offset) << "" << std::setw(length - offset) << std::left << id;

    pcout << std::endl;
  }

  void
  print_own(dealii::ConditionalOStream const & pcout,
            unsigned int const                 offset,
            unsigned int const                 length,
            bool const                         relative = false,
            double const                       ref_time = -1.0) const
  {
    pcout << std::setw(offset) << "" << std::setw(length - offset) << std::left << id;

    dealii::Utilities::MPI::MinMaxAvg ref_time_data =
      dealii::Utilities::MPI::min_max_avg(ref_time, MPI_COMM_WORLD);
    double const ref_time_avg = ref_time_data.avg;

    if(data.get())
    {
      double const time_avg = get_average_wall_time();

      pcout << std::setprecision(precision) << std::scientific << std::setw(10) << std::right
            << time_avg << " s";

      if(relative)
        pcout << std::setprecision(precision) << std::fixed << std::setw(10) << std::right
              << time_avg / ref_time_avg * 100.0 << " %";
    }

    pcout << std::endl;
  }

  void
  print_direct_children(dealii::ConditionalOStream const & pcout,
                        unsigned int const                 offset,
                        unsigned int const                 length,
                        bool const                         relative = false,
                        double const                       ref_time = -1.0) const
  {
    TimerTree other;
    if(relative && sub_trees.size() > 0)
    {
      other.id              = "Other";
      other.data            = std::make_shared<Data>();
      other.data->wall_time = ref_time;
    }

    for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
    {
      if((*it)->data.get())
      {
        (*it)->print_own(pcout, offset, length, relative, ref_time);

        if(relative)
          other.data->wall_time -= (*it)->data->wall_time;
      }
    }

    if(relative && sub_trees.size() > 0)
      other.print_own(pcout, offset, length, relative, ref_time);
  }

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
