#include <mpi.h>
#include <iostream>
#include <string>

#include "../../../src/deal-spectrum.h"
#include "../../../src/util/optional_arguments.h"

int
main(int argc, char ** argv)
{
  using namespace dealspectrum;

  // init MPI...
  MPI_Init(&argc, &argv);
  fftw_mpi_init();
  // ... and get info
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // process arguments: optional arguments ...
  std::map<std::string, std::string> map = processOptionalArguments(argc, argv);
  // ... file to be processed:
  char * filename = argv[1];


  // setup...
  Setup s;
  // ... process file header and set setup...
  if(s.readHeader(filename))
  {
    // ... setup does not match to file header
    MPI_Finalize();
    if(rank == 0)
      std::cout << "Error: File does not match setup!" << std::endl;
    return 1;
  }
  // ... set number of evaluation points if given on command line
  if(map.count("eval"))
    s.points_dst = stoi(map["eval"]);


  // helper classes...
  // ... create and init mapper:
  Bijection h(s);
  // ... create and init interpolator:
  Interpolator ipol(s);
  // ... create and init permuter:
  Permutator fftc(s);
  // ... create and init fftw_wrapper:
  SpectralAnalysis fftw(s);


  // ... initialize
  h.init();
  fftw.init();
  ipol.init(h);
  fftc.init(h, fftw);


  // process file: read file ...
  ipol.deserialize(filename);
  // ... permute values locally and
  ipol.interpolate();
  // ... permute to be row wise
  fftc.ipermute(ipol.dst, fftw.u_real);
  fftc.iwait();


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