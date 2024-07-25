#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


#include "./simplex_grids.h"


/**
 * Adopted from dealii/tests/simplex/matrix_free_02
 * 
 * Works in 2D with p = 1,2,3
 * Quads and triangles use the same Gauss quadrature on the faces (which is in 2D lines)
 * 
 * Does not work in 3D out of the box, need to adopt hp_vertex_dof_identities() and hp_quad_dof_identities() for FE_Q and FE_SimplexP elements.
 * 
 * For pyramids only linear elements are available.  
 */

using namespace dealii;

template <int dim_, int n_components = dim_, typename Number = double>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using number     = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

  using FECellIntegrator = FEEvaluation<dim, -1, 0, n_components, Number>;

  void
  reinit(const hp::MappingCollection<dim>              &mapping,
         const DoFHandler<dim>           &dof_handler,
         const hp::QCollection<dim>           &quad,
         const AffineConstraints<number> &constraints,
         const unsigned int mg_level         = numbers::invalid_unsigned_int,
         const bool         ones_on_diagonal = false)
  {
    this->constraints.copy_from(constraints);

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients;
    data.mg_level             = mg_level;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "Sizes shape info: "
                  << matrix_free.get_shape_info()
                       .data[0]
                       .shape_values.memory_consumption()
                  << " "
                  << matrix_free.get_shape_info()
                       .data[0]
                       .shape_gradients.memory_consumption()
                  << " " << dof_handler.get_fe().dofs_per_cell << " "
                  << matrix_free.get_shape_info().n_q_points << " "
                  << matrix_free.get_shape_info().dofs_per_component_on_cell
                  << " " << matrix_free.get_dof_info(0).dof_indices.size()
                  << " " << dof_handler.get_triangulation().n_active_cells()
                  << " "
                  << dof_handler.get_triangulation().n_global_active_cells()
                  << " " << dof_handler.n_dofs() << std::endl;
        std::cout
          << "Mapping info: "
          << matrix_free.get_mapping_info()
               .cell_data[0]
               .data_index_offsets.size()
          << " n_jacobians "
          << matrix_free.get_mapping_info().cell_data[0].jacobians[0].size()
          << std::endl;
      }

    constrained_indices.clear();

    if (ones_on_diagonal)
      for (auto i : this->matrix_free.get_constrained_dofs())
        constrained_indices.push_back(i);
  }

  virtual types::global_dof_index
  m() const
  {
    if (this->matrix_free.get_mg_level() != numbers::invalid_unsigned_int)
      return this->matrix_free.get_dof_handler().n_dofs(
        this->matrix_free.get_mg_level());
    else
      return this->matrix_free.get_dof_handler().n_dofs();
  }

  Number
  el(unsigned int, unsigned int) const
  {
    DEAL_II_NOT_IMPLEMENTED();
    return 0;
  }

  virtual void
  initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }

  virtual void
  vmult(VectorType &dst, const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &Operator::do_cell_integral_range, this, dst, src, true);

    //for (unsigned int i = 0; i < constrained_indices.size(); ++i)
    //  dst.local_element(constrained_indices[i]) =
    //    src.local_element(constrained_indices[i]);
  }

  void
  Tvmult(VectorType &dst, const VectorType &src) const
  {
    vmult(dst, src);
  }

private:
  void
  do_cell_integral_global(FECellIntegrator &integrator,
                          VectorType       &dst,
                          const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }

  void
  do_cell_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);
   
    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);
        do_cell_integral_global(integrator, dst, src);
     }
  }

  MatrixFree<dim, number> matrix_free;

  AffineConstraints<number> constraints;

  std::vector<unsigned int> constrained_indices;
};


template <int dim, typename Number>
void
do_test(const unsigned int fe_degree)
{
   ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                             0);
  pcout << "Running in " << dim << "D with degree " << fe_degree << std::endl;


  Triangulation<dim> tria;

  const unsigned int subdivisions = dim == 2 ? 25 : 8;
  
  if (dim == 2)
    GridGenerator::subdivided_hyper_cube_with_simplices_mix(tria, subdivisions);
  else
    GridGenerator::cube_and_pyramid_and_tet(tria);

  FE_SimplexP<dim>      fe1(fe_degree);
  FE_Q<dim>             fe2(fe_degree);
  hp::FECollection<dim> fes(fe1, fe2);
  if (dim == 3)
  {
    fes.push_back(FE_PyramidP<dim>(fe_degree));
    fes.push_back(FE_WedgeP<dim>(fe_degree));
  }

  QGaussSimplex<dim>   quad1(fe_degree + 1);
  QGauss<dim>          quad2(fe_degree + 1);
  hp::QCollection<dim> quads(quad1, quad2);
  if (dim == 3)
  {
    quads.push_back(QGaussPyramid<dim>(fe_degree + 1));
    quads.push_back(QGaussWedge<dim>(fe_degree + 1));
  }

  MappingFE<dim>             mapping1(FE_SimplexP<dim>(1));
  MappingQ<dim>              mapping2(1);
  hp::MappingCollection<dim> mappings(mapping1, mapping2);
  if (dim == 3)
  {
    mappings.push_back(MappingFE<dim>(FE_PyramidP<dim>(1)));
    mappings.push_back(MappingFE<dim>(FE_WedgeP<dim>(1)));
  }


  DoFHandler<dim> dof_handler(tria);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->reference_cell() == ReferenceCells::Triangle ||
        cell->reference_cell() == ReferenceCells::Tetrahedron)
      cell->set_active_fe_index(0);
    else if (cell->reference_cell().is_hyper_cube())
      cell->set_active_fe_index(1);
    else if (cell->reference_cell() == ReferenceCells::Pyramid)
      cell->set_active_fe_index(2);
    else if (cell->reference_cell() == ReferenceCells::Wedge)
      cell->set_active_fe_index(3);
    else
      DEAL_II_ASSERT_UNREACHABLE();

  dof_handler.distribute_dofs(fes);

  AffineConstraints<double> constraints;
  DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints);
  constraints.close();


  Operator<dim, 1, Number> op;
      // set up operator
  op.reinit(mappings,
            dof_handler,
            quads,
            constraints,
            numbers::invalid_unsigned_int,
            true);
  LinearAlgebra::distributed::Vector<Number> vec1, vec2;
  op.initialize_dof_vector(vec1);
  for (Number &a : vec1)
    a = static_cast<double>(rand()) / RAND_MAX;
  op.initialize_dof_vector(vec2);

  MPI_Barrier(MPI_COMM_WORLD);
      for (unsigned int r = 0; r < 5; ++r)
        {
          Timer time;
          for (unsigned int t = 0; t < 100; ++t)
          {
            op.vmult(vec2, vec1);
            constraints.distribute(vec2);
          }
          const double run_time = time.wall_time();
          pcout << "n_dofs " << dof_handler.n_dofs() << "  time "
                << run_time / 100 << "  GDoFs/s "
                << 1e-9 * dof_handler.n_dofs() * 100 / run_time << std::endl;
        }
      pcout << "Verification: " << vec2.l2_norm() << std::endl;
      pcout << std::endl;

  

#if 0
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    x.update_ghost_values();
    data_out.add_data_vector(dof_handler, x, "solution");
    data_out.build_patches(mappings, 2);
    data_out.write_vtu_with_pvtu_record("./", "result", 0, MPI_COMM_WORLD);
#endif

 
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  int degree = 1;
  int dim    = 2;
  if (argc > 1)
    dim = std::atoi(argv[1]);
  if (argc > 2)
    degree = std::atoi(argv[2]);

  if (dim == 2)
    do_test<2, double>(degree);
  else
    do_test<3, double>(degree);
}
