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

#ifndef DEAL_SPECTRUM_TIMER
#define DEAL_SPECTRUM_TIMER

namespace dealspectrum
{
/**
 * Class for timing
 */
class DealSpectrumTimer
{
  /**
   * Timing instance assigned to each timing label
   *
   * TODO: MPI, statistics, ...
   */
  class Instance
  {
  public:
    // ... timing instance has been started
    bool started;
    // ... current timing
    clock_t time;
    // ... temporal timing
    clock_t temp;
    // ... how often has this been timed
    int count;
  };

public:
  /**
   * Constructor
   */
  DealSpectrumTimer()
  {
  }

  /**
   * start timing
   *
   * @param label label assigned to timing instance
   */
  void
  start(std::string label)
  {
    if(not m.count(label))
      m[label] = Instance();
    m[label].temp    = clock();
    m[label].started = true;
  }

  /**
   * stop timing: overwrite old timing with new timing
   *
   * @param label label assigned to timing instance
   */
  void
  stop(std::string label)
  {
    if(m.count(label) and m[label].started)
    {
      auto & t  = m[label];
      t.time    = clock() - m[label].temp;
      t.started = false;
      t.count++;
    }
  }

  /**
   * stop timing: add new timing to new timing
   *
   * @param label label assigned to timing instance
   */
  void
  append(std::string label)
  {
    if(m.count(label) and m[label].started)
    {
      auto & t = m[label];
      t.time += (clock() - t.temp);
      t.started = false;
      t.count++;
    }
  }

  /**
   * write timing statistics to screen
   *
   * TODO: write to stream
   */
  void
  printTimings()
  {
    double sum_total   = 0;
    double sum_latency = 0;
    printf("\n\ndeal.spectrum timing statistics in seconds:\n");
    printf("===============================================================\n");
    printf("                      #nr    ave                 sum\n");
    printf("===============================================================\n");
    for(auto const & i : m)
    {
      printf("  %-18s %4d %18.12f %18.12f\n",
             i.first.c_str(),
             i.second.count,
             ((double)i.second.time) / CLOCKS_PER_SEC / i.second.count,
             ((double)i.second.time) / CLOCKS_PER_SEC);
      sum_total += ((double)i.second.time) / CLOCKS_PER_SEC;
      sum_latency += ((double)i.second.time) / CLOCKS_PER_SEC / i.second.count;
    }
    printf("===============================================================\n");
    printf("                          %18.12f %18.12f\n", sum_latency, sum_total);
    printf("===============================================================\n");
    printf("\n\n");
  }

  // map containing timing instances
  std::map<std::string, Instance> m;
};

} // namespace dealspectrum

#endif
