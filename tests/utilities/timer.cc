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
#include <fstream>
#include <iostream>
#include <sstream>

// deal.II
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/utilities/timer_tree.h>



void
test1()
{
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // clang-format off
  pcout << std::endl << std::endl<< std::endl
        << "_____________________________________________________________"<< std::endl
        << "                                                             "<< std::endl
        << "                  Timer: test 1 (basic)                      "<< std::endl
        << "_____________________________________________________________"<< std::endl
        << std::endl;
  // clang-format on

  dealii::Timer timer;
  timer.restart();

  ExaDG::TimerTree tree;

  for(unsigned int i = 0; i < 1000; ++i)
  {
    tree.insert({"General"}, 20.0);

    tree.insert({"General", "Part 1"}, 2.0);

    tree.insert({"General", "Part 2"}, 3.0);

    tree.insert({"General", "Part 2", "Sub a"}, 0.75);

    tree.insert({"General", "Part 2", "Sub b"}, 0.9);

    tree.insert({"General", "Part 3"}, 4.0);

    tree.insert({"General", "Part 3", "Sub a"}, 0.5);

    tree.insert({"General", "Part 3", "Sub a", "sub-sub a"}, 0.04);

    tree.insert({"General", "Part 3", "Sub b"}, 0.98765);

    tree.insert({"General"}, 2.0);
  }


  double wall_time = timer.wall_time();

  if(false)
  {
    pcout << "Wall time for filling the tree = " << std::scientific << wall_time << std::endl
          << std::endl;
    pcout << "Wall time for filling one item of the tree = " << std::scientific << wall_time / 10000
          << std::endl
          << std::endl;
  }

  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings for level = 3:" << std::endl;
  tree.print_level(pcout, 3);
  pcout << std::endl << "timings for level = 4:" << std::endl;
  tree.print_level(pcout, 4);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);

  tree.clear();

  tree.insert({"General", "Part 2", "Sub a"}, 1.234);
  tree.insert({"General", "Part 3", "Sub a"}, 0.123);
  tree.insert({"General", "Part 4"}, 5.678);

  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  // relative timings for level 1 are not possible
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);

  // now, enable relative timings for level 1
  tree.insert({"General"}, 10.0);

  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
}

void
test2()
{
  dealii::ConditionalOStream pcout(std::cout,
                                   dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  // clang-format off
  pcout << std::endl << std::endl<< std::endl
        << "_____________________________________________________________"<< std::endl
        << "                                                             "<< std::endl
        << "                  Timer: test 2 (modular coupling)           "<< std::endl
        << "_____________________________________________________________"<< std::endl
        << std::endl;
  // clang-format on

  dealii::Timer timer;
  timer.restart();

  ExaDG::TimerTree tree;
  tree.insert({"FSI"}, 100.);

  std::shared_ptr<ExaDG::TimerTree> tree_fluid;
  tree_fluid.reset(new ExaDG::TimerTree());
  tree_fluid->insert({"Fluid"}, 70.);
  tree_fluid->insert({"Fluid", "Pressure Poisson"}, 40.);
  tree_fluid->insert({"Fluid", "Postprocessing"}, 10.);
  tree_fluid->insert({"Fluid", "ALE update"}, 15.);


  std::shared_ptr<ExaDG::TimerTree> tree_structure;
  tree_structure.reset(new ExaDG::TimerTree());
  tree_structure->insert({"Structure", "Right-hand side"}, 2.);
  tree_structure->insert({"Structure", "Assemble"}, 9.);
  tree_structure->insert({"Structure", "Solve"}, 14.);
  tree_structure->insert({"Structure"}, 25.);

  tree.insert({"FSI"}, tree_fluid);
  tree.insert({"FSI"}, tree_structure, "Structure");

  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);

  tree.clear();

  // should be empty after clear()
  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree.print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree.print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree.print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree.print_plain(pcout);

  // clear() must no touch sub-trees
  pcout << std::endl << "timings for level = 0:" << std::endl;
  tree_structure->print_level(pcout, 0);
  pcout << std::endl << "timings for level = 1:" << std::endl;
  tree_structure->print_level(pcout, 1);
  pcout << std::endl << "timings for level = 2:" << std::endl;
  tree_structure->print_level(pcout, 2);
  pcout << std::endl << "timings all:" << std::endl;
  tree_structure->print_plain(pcout);
}

int
main(int argc, char ** argv)
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    test1();

    test2();
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
