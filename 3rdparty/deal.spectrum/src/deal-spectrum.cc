#include <mpi.h>
#include <iostream>
#include <limits>

#include "deal-spectrum.h"
#include "util/optional_arguments.h"

int
main(int argc, char ** argv)
{
  using namespace dealspectrum;
  
  MPI_Comm comm = MPI_COMM_WORLD;
  
  // init MPI...
  MPI_Init(&argc, &argv);
  fftw_mpi_init();
  // ... and get info
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // process arguments: optional arguments ...
  std::map<std::string, std::string> map = processOptionalArguments(argc, argv);
  if(map.count("h"))
  {
    if(rank == 0)
    {
      printf("Usage: ./deal-spectrum [OPTION]... [FILE]...\n");
      printf("Process specified files.\n");
      printf("\nOptional arguments:\n");
      printf("   -e        number of evaluation points per cell\n");
      printf("   -l        energy cut-off\n");

      printf("\nOptional flags:\n");
      printf("  --h        print this help page \n");
      printf("  --r        deserialize row-wise format \n");

      printf("\nExample:\n");
      printf("  ./deal-specturm f1 f2 f3          Process file f1, f2, and f3\n");
      printf("  ./deal-specturm -e 10 f           Process file f with 10 evaluation points\n");

      printf("\nMPI-Example:\n");
      printf("  mpirun -np 4 ./deal-specturm f    Process file f with 4 MPI-processes\n");
    }
    MPI_Finalize();
    return 0;
  }

  // Set default configuration
  int    evaluation_points = 0;
  bool   write_row_wise    = false;
  double lower_limit       = std::numeric_limits<double>::min();
  // ... overwrite default configurations with command line provided arguments
  if(map.count("e"))
    evaluation_points = stoi(map["e"]);
  if(map.count("r"))
    write_row_wise = true;
  if(map.count("l"))
    lower_limit = stof(map["l"]);

  // setup...
  Setup s(comm);
  // ... mapper
  Bijection h(comm, s);
  // ... interpolator
  Interpolator ipol(comm, s);
  // ... permuter
  Permutator fftc(comm, s);
  // ... fft-wrapper
  SpectralAnalysis fftw(comm, s);

  // loop over all provided files...
  for(int i = 1; i < argc; i++)
  {
    char * filename = argv[i];
    if(rank == 0)
      std::cout << "Process: " << filename << std::endl;

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
      printf("  Energy (domain):   %20.12f\n", e_d);
      printf("  Energy (spectral): %20.12f\n\n", e_s);
      printf("  Bin   Wave length          Count   Energy\n");
      for(int i = 0; i < len; i++)
        if(E[i] > lower_limit /*|| lower_limit == std::numeric_limits<double>::min()*/)
          printf("%5d %20.12e %7d %20.12e\n", i, kappa[i], (int)C[i], E[i]);
    }
  }
}
