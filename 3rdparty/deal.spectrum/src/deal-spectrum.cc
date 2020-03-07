#include <mpi.h>
#include <iostream>
#include <limits>

#include "deal-spectrum.h"

int
main(int argc, char ** argv)
{
  using namespace dealspectrum;

  MPI_Comm comm = MPI_COMM_WORLD;
  
  MPI_Init(&argc, &argv);
  
  fftw_mpi_init();
  
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  
  if(argc==1)
  {
    if(rank==0)
      printf("Not enough command-line arguments provided!\n");    
    MPI_Finalize();
    return 0;
  }

  Setup s(comm);
  s.cells      = atoi(argv[1]); // number of dofs in each direction
  s.points_dst = 1;  
  s.dim        = 3;
  s.rank       = rank;
  s.size       = size;
  
  SpectralAnalysis fftw(comm, s);

  fftw.init();
  fftw.execute();
  fftw.calculate_energy_spectrum();
  fftw.calculate_energy();
  
  if(rank == 0)
  {
    double * kappa;
    double * E;
    double * C;
    double   e_d;
    double   e_s;
    int      len = fftw.get_results(kappa, E, C, e_d, e_s);
    
    printf("Calculate kinetic energy spectrum at time t = %f\n", s.time);
    printf("  Energy physical space e_phy = %20.12e\n", e_d);
    printf("  Energy spectral space e_spe = %20.12e\n", e_s);
    printf("  Difference  |e_phy - e_spe| = %20.12e\n\n", std::abs(e_s-e_d));
    printf("    k  k (avg)              E(k)\n");
    for(int i = 0; i < len; i++)
      //if(E[i] > std::numeric_limits<double>::min())
        printf("%5d %19.12e %19.12e\n", i, kappa[i], E[i]);
  }
  
  MPI_Finalize();
}
