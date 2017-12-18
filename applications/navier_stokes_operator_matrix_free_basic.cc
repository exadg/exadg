#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>


using namespace dealii;

const types::global_dof_index target_problem_size = 1000000;


template <int dim, int degree, typename Number=double>
class NavierStokesTest
{
public:
  NavierStokesTest(const DoFHandler<dim> &dof_handler,
                   const bool do_renumber)
    :
    dof_handler(dof_handler),
    mapping(dof_handler.get_fe().degree < 10 ? dof_handler.get_fe().degree : 9),
    time_step(1.),
    viscosity(1.),
    tau_projection(1.)
  {
    typename MatrixFree<dim,Number>::AdditionalData mf_data;
    mf_data.mapping_update_flags_inner_faces = update_jacobians | update_normal_vectors;
    mf_data.mapping_update_flags_boundary_faces = update_jacobians | update_normal_vectors;
    ConstraintMatrix constraints;

    if (do_renumber)
      {
        mf_data.initialize_mapping = false;

        std::vector<types::global_dof_index> renumbering;

        matrix_free.reinit(mapping, dof_handler, constraints,
                           QGauss<1>(dof_handler.get_fe().degree+1), mf_data);

        matrix_free.renumber_dofs(renumbering);
        const_cast<DoFHandler<dim> &>(dof_handler).renumber_dofs(renumbering);

        mf_data.initialize_mapping = true;
      }

    matrix_free.reinit(mapping, dof_handler, constraints,
                       QGauss<1>(dof_handler.get_fe().degree+1), mf_data);
  }

  void set_time_step(const double time_step)
  {
    this->time_step = time_step;
  }

  void set_viscosity(const double viscosity)
  {
    this->viscosity = viscosity;
  }

  const MatrixFree<dim, Number> &  get_matrix_free() const
  {
    return matrix_free;
  }

  void helmholtz (LinearAlgebra::distributed::Vector<Number> &dst,
                  const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    matrix_free.loop(&NavierStokesTest::helmholtz_cell,
                     &NavierStokesTest::helmholtz_face,
                     &NavierStokesTest::helmholtz_boundary,
                     this, dst, src, /*zero_dst = */true,
                     MatrixFree<dim,Number>::values_and_gradients,
                     MatrixFree<dim,Number>::values_and_gradients);
  }

  void projection (LinearAlgebra::distributed::Vector<Number> &dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    matrix_free.loop(&NavierStokesTest::project_cell,
                     &NavierStokesTest::project_face,
                     &NavierStokesTest::project_boundary,
                     this, dst, src, /*zero_dst = */true,
                     MatrixFree<dim,Number>::only_values,
                     MatrixFree<dim,Number>::only_values);
  }

private:

  void helmholtz_cell(const MatrixFree<dim,Number> &,
                      LinearAlgebra::distributed::Vector<Number> &dst,
                      const LinearAlgebra::distributed::Vector<Number> &src,
                      const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,degree,degree+1,dim,Number> phi(matrix_free);
    VectorizedArray<Number> factor = make_vectorized_array<Number>(time_step * viscosity);

    for (unsigned int cell=cell_range.first; cell!=cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        for (unsigned int q=0; q<phi.static_n_q_points; ++q)
          {
            phi.submit_value(phi.get_value(q), q);
            phi.submit_gradient(factor * phi.get_gradient(q), q);
          }
        phi.integrate_scatter(true, true, dst);
      }
  }

  void helmholtz_face(const MatrixFree<dim,Number> &,
                      LinearAlgebra::distributed::Vector<Number> &dst,
                      const LinearAlgebra::distributed::Vector<Number> &src,
                      const std::pair<unsigned int,unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim,degree,degree+1,dim,Number> fe_eval(matrix_free,true);
    FEFaceEvaluation<dim,degree,degree+1,dim,Number> fe_eval_neighbor(matrix_free,false);
    typedef typename FEFaceEvaluation<dim,degree,degree+1,dim,Number>::value_type value_type;
    for (unsigned int face=face_range.first; face!=face_range.second; ++face)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        fe_eval.gather_evaluate(src,true,true);
        fe_eval_neighbor.gather_evaluate(src,true,true);

        VectorizedArray<Number> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
                                          std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
          (Number)(degree * (degree + 1.0)) * 0.5 * 2.0;

        for(unsigned int q=0;q<fe_eval.static_n_q_points;++q)
          {
            value_type half_jump = (fe_eval.get_value(q)-
                                    fe_eval_neighbor.get_value(q)) * make_vectorized_array<Number>(0.5);
            value_type average_valgrad =
              (fe_eval.get_normal_gradient(q) +
               fe_eval_neighbor.get_normal_gradient(q)) * make_vectorized_array<Number>(0.5);
            average_valgrad = half_jump * sigmaF - average_valgrad;

            fe_eval.submit_normal_gradient(-half_jump,q);
            fe_eval_neighbor.submit_normal_gradient(-half_jump,q);
            fe_eval.submit_value(average_valgrad,q);
            fe_eval_neighbor.submit_value(-average_valgrad,q);
          }

        fe_eval.integrate_scatter(true,true,dst);
        fe_eval_neighbor.integrate_scatter(true,true,dst);
      }
  }

  void helmholtz_boundary(const MatrixFree<dim,Number> &,
                          LinearAlgebra::distributed::Vector<Number> &dst,
                          const LinearAlgebra::distributed::Vector<Number> &src,
                          const std::pair<unsigned int,unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim,degree,degree+1,dim,Number> fe_eval(matrix_free,true);
    typedef typename FEFaceEvaluation<dim,degree,degree+1,dim,Number>::value_type value_type;

    for (unsigned int face=face_range.first; face!=face_range.second; ++face)
      {
        fe_eval.reinit (face);

        fe_eval.gather_evaluate(src,true,true);

        VectorizedArray<Number> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction())) *
          (Number)(degree * (degree + 1.0)) * 2.0;

        for(unsigned int q=0;q<fe_eval.static_n_q_points;++q)
          {
            value_type half_jump = fe_eval.get_value(q);
            value_type average_valgrad = -fe_eval.get_normal_gradient(q);
            fe_eval.submit_normal_gradient(-half_jump,q);
            average_valgrad += half_jump * sigmaF;
            fe_eval.submit_value(average_valgrad,q);
          }

        fe_eval.integrate_scatter(true,true,dst);
      }
  }

  void project_cell(const MatrixFree<dim,Number> &,
                    LinearAlgebra::distributed::Vector<Number> &dst,
                    const LinearAlgebra::distributed::Vector<Number> &src,
                    const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,degree,degree+1,dim,Number> phi(matrix_free);
    VectorizedArray<Number> factor = make_vectorized_array<Number>(time_step * viscosity);

    for (unsigned int cell=cell_range.first; cell!=cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(src, true, true);
        for (unsigned int q=0; q<phi.static_n_q_points; ++q)
          {
            phi.submit_value(phi.get_value(q), q);
            phi.submit_divergence(tau_projection * phi.get_divergence(q), q);
          }
        phi.integrate_scatter(true, true, dst);
      }
  }

  void project_face(const MatrixFree<dim,Number> &,
                    LinearAlgebra::distributed::Vector<Number> &dst,
                    const LinearAlgebra::distributed::Vector<Number> &src,
                    const std::pair<unsigned int,unsigned int> &face_range) const
  {
    FEFaceEvaluation<dim,degree,degree+1,dim,Number> fe_eval(matrix_free,true);
    FEFaceEvaluation<dim,degree,degree+1,dim,Number> fe_eval_neighbor(matrix_free,false);
    typedef typename FEFaceEvaluation<dim,degree,degree+1,dim,Number>::value_type value_type;
    for (unsigned int face=face_range.first; face!=face_range.second; ++face)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        fe_eval.gather_evaluate(src,true,false);
        fe_eval_neighbor.gather_evaluate(src,true,false);

        for(unsigned int q=0;q<fe_eval.static_n_q_points;++q)
          {
            Tensor<1,dim,VectorizedArray<Number> > normal = fe_eval.get_normal_vector(q);
            VectorizedArray<Number> jump_val = normal * (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q));
            fe_eval.submit_value(normal * jump_val, q);
            fe_eval.submit_value(-normal * jump_val, q);
          }

        fe_eval.integrate_scatter(true,false,dst);
        fe_eval_neighbor.integrate_scatter(true,false,dst);
      }
  }

  void project_boundary(const MatrixFree<dim,Number> &,
                        LinearAlgebra::distributed::Vector<Number> &dst,
                        const LinearAlgebra::distributed::Vector<Number> &src,
                        const std::pair<unsigned int,unsigned int> &face_range) const
  {
  }

  const DoFHandler<dim> &dof_handler;
  const MappingQGeneric<dim> mapping;
  double time_step;
  double viscosity;
  double tau_projection;
  MatrixFree<dim,Number> matrix_free;
};


template <int dim, int degree>
void do_test(const FiniteElement<dim> &fe_scalar,
             const bool do_renumber = true)
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, 1);
  while (tria.n_global_active_cells() * fe_scalar.dofs_per_cell * dim <
         target_problem_size)
    tria.refine_global(1);
  FESystem<dim> fe(fe_scalar, dim);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
  pcout << "Type of element:              " << fe_scalar.get_name() << std::endl;
  pcout << "Renumber degrees of freedom:  " << do_renumber << std::endl;
  pcout << "Number of cells:              " << tria.n_global_active_cells() << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

  {
    NavierStokesTest<dim,degree,double> test(dof_handler, do_renumber);
    LinearAlgebra::distributed::Vector<double> v1, v2;
    test.get_matrix_free().initialize_dof_vector(v1);
    test.get_matrix_free().initialize_dof_vector(v2);
    v1 = 1;

    Timer time;
    double min_time = 1e10;
    for (unsigned int i=0; i<5; ++i)
      {
        time.restart();
        for (unsigned int t=0; t<100; ++t)
          test.helmholtz(v2, v1);
        Utilities::MPI::MinMaxAvg data =
          Utilities::MPI::min_max_avg (time.wall_time()/100, MPI_COMM_WORLD);
        pcout << "Helmholtz mat-vec dp  "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << std::endl;
        min_time = std::min(min_time, data.max);
      }
    pcout << "Helmholtz dp deg=" << degree << " dofs=" << dof_handler.n_dofs() << " DoFs/s: "
          << dof_handler.n_dofs() / min_time << std::endl;
    pcout << std::endl;

    min_time = 1e10;
    for (unsigned int i=0; i<5; ++i)
      {
        time.restart();
        for (unsigned int t=0; t<100; ++t)
          test.projection(v2, v1);
        Utilities::MPI::MinMaxAvg data =
          Utilities::MPI::min_max_avg (time.wall_time()/100, MPI_COMM_WORLD);
        pcout << "Projection mat-vec dp "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << std::endl;
        min_time = std::min(min_time, data.max);
      }
    pcout << "Projection dp deg=" << degree << " dofs=" << dof_handler.n_dofs() << " DoFs/s: "
          << dof_handler.n_dofs() / min_time << std::endl;
    pcout << std::endl;
  }

  {
    NavierStokesTest<dim,degree,float> test(dof_handler, do_renumber);
    LinearAlgebra::distributed::Vector<float> v1, v2;
    test.get_matrix_free().initialize_dof_vector(v1);
    test.get_matrix_free().initialize_dof_vector(v2);
    v1 = 1;

    Timer time;
    double min_time = 1e10;
    for (unsigned int i=0; i<5; ++i)
      {
        time.restart();
        for (unsigned int t=0; t<100; ++t)
          test.helmholtz(v2, v1);
        Utilities::MPI::MinMaxAvg data =
          Utilities::MPI::min_max_avg (time.wall_time()/100, MPI_COMM_WORLD);
        pcout << "Helmholtz mat-vec sp  "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << std::endl;
        min_time = std::min(min_time, data.max);
      }
    pcout << "Helmholtz sp deg=" << degree << " dofs=" << dof_handler.n_dofs() << " DoFs/s: "
          << dof_handler.n_dofs() / min_time << std::endl;
    pcout << std::endl;

    min_time = 1e10;
    for (unsigned int i=0; i<5; ++i)
      {
        time.restart();
        for (unsigned int t=0; t<100; ++t)
          test.projection(v2, v1);
        Utilities::MPI::MinMaxAvg data =
          Utilities::MPI::min_max_avg (time.wall_time()/100, MPI_COMM_WORLD);
        pcout << "Projection mat-vec sp "
              << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")" << std::endl;
        min_time = std::min(min_time, data.max);
      }
    pcout << "Projection sp deg=" << degree << " dofs=" << dof_handler.n_dofs() << " DoFs/s: "
          << dof_handler.n_dofs() / min_time << std::endl;
    pcout << std::endl;
  }
}


int main(int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl << std::endl;

      std::cout << std::endl
                << "Number of MPI ranks:         "
                << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                << std::endl << std::endl;
    }

  const unsigned int dim = 3;
  do_test<dim,2>(FE_DGQ<dim>(2), false);
  do_test<dim,4>(FE_DGQ<dim>(4), false);
  do_test<dim,5>(FE_DGQ<dim>(5), false);
  do_test<dim,4>(FE_DGQHermite<dim>(4), false);
  do_test<dim,5>(FE_DGQHermite<dim>(5), false);
  do_test<dim,4>(FE_DGQHermite<dim>(4), true);
  do_test<dim,5>(FE_DGQHermite<dim>(5), true);
}
