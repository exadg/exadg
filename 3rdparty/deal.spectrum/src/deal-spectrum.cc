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
    char * filename = argv[i];

    // read file header and check it...
    if(s.readHeader(filename))
    {
      // ... setup does not match to file header
      MPI_Finalize();
      if(rank == 0)
        std::cout << "Error: File does not match setup!" << std::endl;
      return 1;
    }

    // setup data for fftw and read input file...
    if(s.type == 0)
    {
      // overwrite number of evaluation points
      if(evaluation_points > 0)
        s.points_dst = evaluation_points;
      // ... cell wise along sfc -> init mapping
      h.init();
      fftw.init();
      ipol.init(h);
      fftc.init(h, fftw);
      // ... read dofs from file
      ipol.deserialize(filename);
      // ... interpolate values
      ipol.interpolate();
      // ... permute to be row wise
      fftc.ipermute(ipol.dst, fftw.u_real);
      fftc.iwait();

      if(write_row_wise)
      {
        // write permuted data to file: create new file name...
        std::string a(filename);
        a.append("_converted");
        // ... write header
        Setup s_ = s;
        s_.type  = 1;
        s_.writeHeader(a.c_str());
        // ... write data
        fftw.serialize(a.c_str());
      }
    }
    else if(s.type == 1)
    {
      // ... row wise
      fftw.init();
      fftw.deserialize(filename);
    }
    else
    {
      // ... file type is not supported
      MPI_Finalize();
      if(rank == 0)
        std::cout << "Error: File type is not supported (choose: 0, 1)!" << std::endl;
      return 1;
    }

    // perform FFT & post process...
    fftw.execute();
    fftw.calculate_energy_spectrum();
    fftw.calculate_energy();

    if(rank == 0)
    {
      // ... and print results
      double * kappa;
      double * E;
      double * C;
      double   e_d;
      double   e_s;
      int      len = fftw.get_results(kappa, E, C, e_d, e_s);
      
      printf("\nCalculate kinetic energy spectrum at time t = %f:\n\n", s.time);
      printf("  Energy physical space e_phy = %18.12e\n", e_d);
      printf("  Energy spectral space e_spe = %18.12e\n", e_s);
      printf("  Difference  |e_phy - e_spe| = %18.12e\n\n", std::abs(e_s-e_d));
      printf("    k  k (avg)              E(k)\n");
      for(int i = 0; i < len; i++)
        if(E[i] > lower_limit /*|| lower_limit == std::numeric_limits<double>::min()*/)
          printf("%5d %19.12e %20.12e\n", i, kappa[i], E[i]);
    }
  }

  MPI_Finalize();
}
