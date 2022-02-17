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

// C++
#include <iomanip>

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
TimerTree::TimerTree() : id("")
{
}

void
TimerTree::clear()
{
  this->id = "";
  data     = nullptr;
  sub_trees.clear();
}

void
TimerTree::insert(std::vector<std::string> const ids, double const wall_time)
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

void
TimerTree::insert(std::vector<std::string>   ids,
                  std::shared_ptr<TimerTree> sub_tree,
                  std::string const          new_name)
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

void
TimerTree::print_plain(dealii::ConditionalOStream const & pcout) const
{
  unsigned int const length = get_length();

  pcout << std::endl;

  do_print_plain(pcout, 0, length);
}

void
TimerTree::print_level(dealii::ConditionalOStream const & pcout, unsigned int const level) const
{
  unsigned int const length = get_length();

  unsigned int const max_level = get_max_level();

  if(level <= max_level)
  {
    pcout << std::endl;

    do_print_level(pcout, level, 0, length);
  }
  else
  {
    pcout << std::endl
          << "Timings can not be printed for level = " << level << "," << std::endl
          << "since the maximum level of the timer tree is max_level = " << max_level << "."
          << std::endl;
  }
}

unsigned int
TimerTree::get_max_level() const
{
  unsigned int max_level = 0;

  for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
  {
    // add + 1 since we only arrive here if a sub-tree exists, i.e.
    // if there is at least one more level
    max_level = std::max(max_level, (*it)->get_max_level() + 1);
  }

  return max_level;
}

void
TimerTree::copy_from(std::shared_ptr<TimerTree> other)
{
  *this = *other;
}

std::vector<std::string>
TimerTree::erase_first(std::vector<std::string> const & in) const
{
  AssertThrow(in.size() > 0, dealii::ExcMessage("empty name."));

  std::vector<std::string> out(in);
  out.erase(out.begin());

  return out;
}

double
TimerTree::get_average_wall_time() const
{
  dealii::Utilities::MPI::MinMaxAvg time_data =
    dealii::Utilities::MPI::min_max_avg(data->wall_time, MPI_COMM_WORLD);

  return time_data.avg;
}

unsigned int
TimerTree::get_length() const
{
  unsigned int length = id.length();

  for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
  {
    length = std::max(length, (*it)->get_length() + offset_per_level);
  }

  return length;
}

void
TimerTree::do_print_plain(dealii::ConditionalOStream const & pcout,
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
TimerTree::do_print_level(dealii::ConditionalOStream const & pcout,
                          unsigned int const                 level,
                          unsigned int const                 offset,
                          unsigned int const                 length) const
{
  if(id.empty())
    return;

  if(level == 0)
  {
    print_own(pcout, offset, length);
  }
  else if(level == 1)
  {
    if(sub_trees.size() > 0)
    {
      if(data.get())
      {
        print_own(pcout, offset, length, true, data->wall_time);
        print_direct_children(pcout, offset + offset_per_level, length, true, data->wall_time);
      }
      else
      {
        print_name(pcout, offset, length, true);
        print_direct_children(pcout, offset + offset_per_level, length);
      }
    }
  }
  else
  {
    // only print name
    print_name(pcout, offset, length, true);

    // recursively print sub trees (decreasing the level and incrementing
    // the offset)
    for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
    {
      (*it)->do_print_level(pcout, level - 1, offset + offset_per_level, length);
    }
  }
}

void
TimerTree::print_name(dealii::ConditionalOStream const & pcout,
                      unsigned int const                 offset,
                      unsigned int const                 length,
                      bool const                         new_line) const
{
  pcout << std::setw(offset) << "" << std::setw(length - offset) << std::left << id;

  if(new_line)
    pcout << std::endl;
}

void
TimerTree::print_own(dealii::ConditionalOStream const & pcout,
                     unsigned int const                 offset,
                     unsigned int const                 length,
                     bool const                         relative,
                     double const                       ref_time) const
{
  print_name(pcout, offset, length, false);

  if(data.get())
  {
    double const time_avg = get_average_wall_time();

    pcout << std::setprecision(precision) << std::scientific << std::setw(10) << std::right
          << time_avg << " s";

    if(relative)
    {
      dealii::Utilities::MPI::MinMaxAvg ref_time_data =
        dealii::Utilities::MPI::min_max_avg(ref_time, MPI_COMM_WORLD);
      double const ref_time_avg = ref_time_data.avg;

      pcout << std::setprecision(precision) << std::fixed << std::setw(10) << std::right
            << time_avg / ref_time_avg * 100.0 << " %";
    }
  }

  pcout << std::endl;
}

void
TimerTree::print_direct_children(dealii::ConditionalOStream const & pcout,
                                 unsigned int const                 offset,
                                 unsigned int const                 length,
                                 bool const                         relative,
                                 double const                       ref_time) const
{
  bool at_least_one_subtree_with_data = false;
  for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
    if((*it)->data.get())
      at_least_one_subtree_with_data = true;

  // Compute item "Other" if relative computation is activated.
  // Print "Other" only if there is at least one sub-tree with data.
  if(relative && at_least_one_subtree_with_data)
  {
    TimerTree other;
    other.id              = "Other";
    other.data            = std::make_shared<Data>();
    other.data->wall_time = ref_time;

    // Print only those sub-trees that contain data.
    // Note that if we would also print sub-trees without data,
    // an interpretation of the meaning of "Other" would not be
    // possible.
    for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
    {
      if((*it)->data.get())
      {
        (*it)->print_own(pcout, offset, length, relative, ref_time);
        other.data->wall_time -= (*it)->data->wall_time;
      }
    }

    other.print_own(pcout, offset, length, relative, ref_time);
  }
  else
  {
    // In this branch we print all sub-trees. If a sub-tree does not
    // contain data, only the name will be printed. Compared to the
    // if-branch above, this is unproblematic since the item "Other"
    // will not be printed.
    for(auto it = sub_trees.begin(); it != sub_trees.end(); ++it)
      (*it)->print_own(pcout, offset, length, relative, ref_time);
  }
}

} // namespace ExaDG
