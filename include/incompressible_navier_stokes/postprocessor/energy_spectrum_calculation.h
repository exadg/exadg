/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_

#include <limits>
#include <vector>
#include <cmath>
#include <memory>

#ifdef USE_DEAL_SPECTRUM
    #include "../../../3rdparty/deal.spectrum/src/deal-spectrum.h"
#else
    namespace dealspectrum{

        class DealSpectrumWrapper{

        public:
            DealSpectrumWrapper(bool , bool ){}

            virtual ~DealSpectrumWrapper() {}

            void init(int , int , int , int , int ){ }

            void execute(const double* ){ }

            int get_results(double*& , double*& , double*& ){ return 0; } 

        };

    }
#endif

template<int dim, int fe_degree, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  KineticEnergySpectrumCalculator()
    :
    clear_files(true),
    dsw(false,true)
  {}

  void setup(MatrixFree<dim,Number> const    &matrix_free_data_in,
             DofQuadIndexData const          &dof_quad_index_data_in,
             KineticEnergySpectrumData const &data_in)
  {
      
    data = data_in;
    
    int local_cells = matrix_free_data_in.n_physical_cells();
    int       cells = local_cells;
    MPI_Allreduce(MPI_IN_PLACE, &cells, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    cells = round(pow(cells,1.0/dim));
    
    //std::cout << local_cells << std::endl; std::cout << cells << std::endl; exit(0);
    
    dsw.init(dim, cells,fe_degree+1,fe_degree+1,local_cells);

  }

  void evaluate(parallel::distributed::Vector<Number> const &velocity,
                double const                                &time,
                int const                                   &time_step_number)
  {
    if(data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
        do_evaluate(velocity,time,time_step_number);
      else // steady problem (time_step_number = -1)
      {
        AssertThrow(false, ExcMessage("Calculation of kinetic energy spectrum only implemented for unsteady problems."));
      }
    }
  }

private:
  void do_evaluate(parallel::distributed::Vector<Number> const &velocity,
                   double const                                ,
                   unsigned int const                          time_step_number)
  {
        if((time_step_number-1)%data.calculate_every_time_steps == 0) {
            if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) 
                std::cout << "Calculate kinetic energy spectrum" << std::endl;
            
            const double& temp = velocity.local_element(0);
            dsw.execute(&temp);

            // write output file
            if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) {
                std::ostringstream filename;
                filename << data.filename_prefix;

                std::ofstream f;
                if(clear_files == true) {
                    f.open(filename.str().c_str(),std::ios::trunc);
                    f << "  Bin  Wave length          Energy" << std::endl;
                    clear_files = false;
                } else {
                    f.open(filename.str().c_str(),std::ios::app);
                }

                // get tabularized results ...
                double* kappa; double* E; double* C;
                int len = dsw.get_results(kappa, E, C);
                
                // ... and print it line by line:
                for(int i = 0; i < len; i++)
                    //if(E[i] > data.output_precision)
                        f << std::scientific << std::setprecision(0) << std::setw(2+ceil(std::max(3.0, log(len)/log(10)))) 
                          << i
                          << std::scientific << std::setprecision(precision) << std::setw(precision+8) 
                          << kappa[i] << "   " << E[i] << std::endl;
            }
        }
  }

  bool clear_files;
  dealspectrum::DealSpectrumWrapper dsw;
  KineticEnergySpectrumData data;
  const unsigned int precision = 12;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_ */
