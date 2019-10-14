
// C++
#include <fstream>
#include <iostream>
#include <sstream>

// deal.ii
#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/partitioner.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/grid/grid_generator.h>

#include "../include/functionalities/calculate_maximum_aspect_ratio.h"

using namespace dealii;

template<int dim>
double
compute_aspect_ratio_hyper_rectangle(Point<dim> const &                left,
                                     Point<dim> const &                right,
                                     std::vector<unsigned int> const & refinements,
                                     unsigned int                      degree     = 1,
                                     unsigned int                      n_q_points = 2,
                                     bool                              deform     = false,
                                     double                            factor     = 1.0)
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(tria, refinements, left, right, false);

  if(deform)
  {
    Tensor<1, dim> diag = right - left;
    double         l    = diag.norm();
    Point<dim>     shift;
    for(unsigned int d = 0; d < dim; ++d)
      shift[d] = l * factor * (0.05 + d * 0.01);
    tria.begin_active()->vertex(0) += shift;
  }

  MappingQGeneric<dim> const mapping(degree);
  QGauss<dim> const          gauss(n_q_points);

  Vector<double> ratios = calculate_aspect_ratio_of_cells(tria, mapping, gauss);
  std::cout << "Aspect ratio vector = ";

  ratios.print(std::cout);

  return calculate_maximum_aspect_ratio(tria, mapping, gauss);
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    // 1D
    {
      std::cout << std::endl << "One dimensional test cases:" << std::endl;

      Point<1> left  = Point<1>(0.0);
      Point<1> right = Point<1>(1.0);

      double ar = 0.0;

      std::vector<unsigned int> refine(1, 2);

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      refine[0] = 5;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;
    }

    // 2D
    {
      std::cout << std::endl << "Two dimensional test cases:" << std::endl;

      Point<2> left  = Point<2>(0.0, 0.0);
      Point<2> right = Point<2>(1.0, 1.0);

      double ar = 0.0;

      std::vector<unsigned int> refine(2, 2);

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true, 10);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      refine[0] = 5;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true, 10);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;
    }

    // 3D
    {
      std::cout << std::endl << "Three dimensional test cases:" << std::endl;

      Point<3> left  = Point<3>(0.0, 0.0, 0.0);
      Point<3> right = Point<3>(1.0, 1.0, 1.0);

      double ar = 0.0;

      std::vector<unsigned int> refine(3, 2);

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true, 10);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      refine[0] = 5;
      refine[1] = 3;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;

      ar = compute_aspect_ratio_hyper_rectangle(left, right, refine, 1, 2, true, 10);
      std::cout << "aspect ratio max    = " << ar << std::endl << std::endl;
    }
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
