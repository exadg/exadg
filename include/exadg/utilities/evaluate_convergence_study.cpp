/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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
#include <filesystem>
#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/mpi.h>

// ExaDG
#include <exadg/utilities/evaluate_convergence_study.h>

namespace ExaDG
{
void
evaluate_convergence_study(MPI_Comm const & mpi_comm, bool const is_test)
{
  if(is_test)
  {
    // convergence study is not included in tests.
  }
  else
  {
    // populate and print convergence study
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      // find data in 'output' folder
      std::string                 output_directory = "output/";
      std::filesystem::path const fs_output_directory{output_directory};
      if(std::filesystem::exists(fs_output_directory))
      {
        // process files in 'output/' starting with 'run_'
        std::string               filename_base = output_directory + "run_";
        std::vector<std::string>  labels;
        std::vector<unsigned int> run_ids;
        std::vector<double>       errors;
        for(auto const & fs_dir_entry : std::filesystem::directory_iterator{fs_output_directory})
        {
          std::string dir_entry = fs_dir_entry.path().string();
          if(dir_entry.size() >= filename_base.size())
          {
            bool base_match = 0 == dir_entry.compare(0, filename_base.size(), filename_base);
            if(base_match)
            {
              std::string run_id_string =
                dir_entry.substr(filename_base.size(), dir_entry.size() - filename_base.size());
              run_id_string = run_id_string.substr(0, run_id_string.find_first_of("_"));
              run_ids.push_back(dealii::Utilities::string_to_int(run_id_string));

              std::string label = dir_entry.substr(filename_base.size() + run_id_string.size() + 1);
              labels.push_back(label);

              // get the final error from the file
              errors.push_back(0.0); // enlarge vector
              std::ifstream infile(dir_entry);
              std::string   line;
              std::getline(infile, line); // skip header line
              while(std::getline(infile, line))
              {
                std::istringstream iss(line);
                double             time, error;
                if(not(iss >> time >> error))
                  AssertThrow(false,
                              dealii::ExcMessage("Could not parse error in file: " + dir_entry));

                errors.back() = error; // overwrite last entry
              }
            }
          }
        }

        // populate and print the convergence table if enough data was found
        unsigned int max_run_id =
          run_ids.size() == 0 ? 0 : *std::max_element(run_ids.begin(), run_ids.end());

        if(labels.size() == 0)
        {
          std::cout
            << "Could not detect any files matching \"output/run_\", no convergence table to display.";
        }
        else if(max_run_id == 0)
        {
          std::cout
            << "Detected files with run_id = 0 matching \"output/run_\", no convergence table to display.";
        }
        else
        {
          // populate the convergence table
          std::vector<std::string> unique_labels;
          dealii::ConvergenceTable convergence_table;
          for(unsigned int run_id = 0; run_id <= max_run_id; ++run_id)
          {
            for(unsigned int i = 0; i < labels.size(); ++i)
            {
              if(run_ids[i] == run_id)
              {
                bool no_match = true;
                for(unsigned int j = 0; j < unique_labels.size(); ++j)
                {
                  if(0 == labels[i].compare(unique_labels[j]))
                  {
                    no_match = false;
                    break;
                  }
                }
                if(no_match)
                {
                  unique_labels.push_back(labels[i]);
                }

                convergence_table.add_value(labels[i] + "  ", errors[i]);
              }
            }
          }

          for(unsigned int i = 0; i < unique_labels.size(); ++i)
          {
            convergence_table.set_scientific(labels[i] + "  ", true);
          }
          convergence_table.evaluate_all_convergence_rates(
            dealii::ConvergenceTable::reduction_rate_log2);

          unsigned int min_run_id = *std::min_element(run_ids.begin(), run_ids.end());
          std::cout << "Convergence table for run_id = " << min_run_id << ", ..., " << max_run_id
                    << ":" << std::endl
                    << std::endl;
          convergence_table.write_text(std::cout);
        }

        std::cout
          << std::endl
          << "_________________________________________________________________________________"
          << std::endl
          << std::endl;
      }
    }
  }
}

} // namespace ExaDG
