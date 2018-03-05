/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn/muench
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_

#include <limits>
#include <vector>
#include <cmath>
#include <memory>

#include "../../../src/deal-spectrum.h"

struct KineticEnergySpectrumData {
    KineticEnergySpectrumData() :
        KineticEnergySpectrumData(false,std::numeric_limits<unsigned int>::max()){}
    
    KineticEnergySpectrumData(bool calculate, unsigned int calculate_every_time_steps) :
        calculate(calculate),
        calculate_every_time_steps(calculate_every_time_steps),
        filename_prefix("energy_spectrum"),
        output_precision(1e-12){}

    void print(ConditionalOStream &pcout){
        if(calculate == true){
            pcout << "  Calculate energy spectrum:" << std::endl;
            //print_parameter(pcout,"Calculate energy spectrum",calculate);
            //print_parameter(pcout,"Calculate every timesteps",calculate_every_time_steps);
        }
    }

    bool calculate;
    unsigned int calculate_every_time_steps;
    std::string filename_prefix;
    double output_precision;
};


template<int dim, int n_global_refinements, int fe_degree, typename Number>
class KineticEnergySpectrumCalculator {
public:
    KineticEnergySpectrumCalculator() : 
        clear_files(true), 
        dsw(false,true) {}

    void setup(
            KineticEnergySpectrumData const &data_in,
            parallel::distributed::Triangulation<dim> & triangulation) {
        data = data_in;
        
        int       cells = Utilities::fixed_int_power<2,n_global_refinements>::value;
        int local_cells = triangulation.n_locally_owned_active_cells();
        
        dsw.init(dim, cells,fe_degree+1,fe_degree+1,local_cells);

    }

    void evaluate(parallel::distributed::Vector<Number> const &velocity,
                  double       time,
                  unsigned int time_step_number){
        if(data.calculate == true)
            do_evaluate(velocity,time,time_step_number);
    }

private:

    void do_evaluate(parallel::distributed::Vector<Number> const &velocity,
                     double       ,
                     unsigned int time_step_number) {

        if((time_step_number-1)%data.calculate_every_time_steps == 0) {
            if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0) 
                std::cout << "Calculate kinetic energy spectrum" << std::endl;
            
            const double* temp = velocity.begin();
            dsw.execute(temp);

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
                    if(E[i] > data.output_precision)
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


#endif 