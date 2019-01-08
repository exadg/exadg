#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/multigrid/mg_base.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/base/mg_level_object.h>

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;

#include "../../../operators/operation-base-util/interpolate.h"
#include "../../../operators/operation-base-util/l2_norm.h"

template <int dim, typename DoFHandlerType = DoFHandler<dim>>
class  MGDataOut : public DataOut<dim, DoFHandlerType>{
    
    public:
    
    MGDataOut(unsigned int level) : level(level){}
    private:
    typename DataOut<dim, DoFHandlerType>::cell_iterator first_cell()
  {
        //return this->triangulation->begin_active();
        
        return this->triangulation->begin(level);
        //typename DoFHandler<dim>::cell_iterator endc=dof_handler.end(level-1);
    }
    
   typename DataOut<dim, DoFHandlerType>::cell_iterator next_cell(
     const typename DataOut<dim, DoFHandlerType>::cell_iterator &cell)
   {
//     // convert the iterator to an active_iterator and advance this to the next
//     // active cell
//     typename Triangulation<DoFHandlerType::dimension,
//                            DoFHandlerType::space_dimension>::active_cell_iterator
//       active_cell = cell;
//     ++active_cell;
//     return active_cell;
     typename Triangulation<DoFHandlerType::dimension,
                            DoFHandlerType::space_dimension>::cell_iterator 	
       active_cell = cell;
     
     if(cell==this->triangulation->end(level))
         return this->dofs->end();
     
     ++active_cell;
     return active_cell;
  }
    
   unsigned int level;
};


template<int dim>
class TestSolution : public Function<dim>
{
public:
  TestSolution(const double time = 0.) : Function<dim>(1, time), wave_number(1.)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const
  {
    return p[0];
    //double result = std::sin(wave_number * p[0] * numbers::PI);
    //for(unsigned int d = 1; d < dim; ++d)
    //  result *= std::sin(wave_number * p[d] * numbers::PI);
    //return result;
  }

private:
  const double wave_number;
};


template<int dim, int fe_degree_1>
class Runner
{
public:
  Runner()
    : triangulation(MPI_COMM_WORLD,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe(fe_degree_1),
      dof_handler(triangulation),
      mapping_1(fe_degree_1),
      quadrature_1(fe_degree_1 + 1),
      global_refinements(dim == 2 ? 4 : 3)
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:

  parallel::distributed::Triangulation<dim> triangulation;
  FE_DGQ<dim>                               fe;
  DoFHandler<dim>                           dof_handler;
  MappingQGeneric<dim>                      mapping_1;

  AffineConstraints<double> dummy_1;

  QGauss<1>                                quadrature_1;
  const unsigned int                       global_refinements;
  
  MGLevelObject<MatrixFree<dim, value_type>> data_1;
  MGLevelObject<VectorType> vectors;
  

  void
  init_triangulation_and_dof_handler()
  {
    const double left        = -1.0;
    const double right       = +1.0;

    GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(global_refinements);

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

  }

  void
  init_boundary_conditions()
  {
      // TODO
  }

  void
  init_matrixfree_and_constraint_matrix()
  {

    dummy_1.clear();

    data_1.resize(0, global_refinements);
    
    for(unsigned int level = 0; level <=global_refinements; level++){
      typename MatrixFree<dim, value_type>::AdditionalData additional_data_1;
      additional_data_1.mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points ;
      additional_data_1.level_mg_handler = level;
      data_1[level].reinit(mapping_1, dof_handler, dummy_1, quadrature_1, additional_data_1);
    }
  }
  
  void 
  init_vectors()
  {

    vectors.resize(0, global_refinements);
    for(unsigned int level = 0; level <=global_refinements; level++)
        data_1[level].initialize_dof_vector(vectors[level]);
      
    MGTools::interpolate(dof_handler, TestSolution<dim>(0), vectors[global_refinements], numbers::invalid_unsigned_int);
    
  }
  
public:
  void
  run()
  {
    // initialize the system
    init_triangulation_and_dof_handler();
    init_boundary_conditions();
    init_matrixfree_and_constraint_matrix();
    init_vectors();
    
    for(unsigned int level = global_refinements; level >=1; level--){
        // TODO: update ghost values
        std::vector<value_type> dof_values_coarse(fe.dofs_per_cell);
        Vector<value_type> dof_values_fine(fe.dofs_per_cell);
        Vector<value_type> tmp(fe.dofs_per_cell);
        std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
        typename DoFHandler<dim>::cell_iterator cell=dof_handler.begin(level-1);
        typename DoFHandler<dim>::cell_iterator endc=dof_handler.end(level-1);
        for ( ; cell != endc; ++cell)
          if (cell->is_locally_owned_on_level())
            {
              Assert(cell->has_children(), ExcNotImplemented());
              std::fill(dof_values_coarse.begin(), dof_values_coarse.end(), 0.);
              for (unsigned int child=0; child<cell->n_children(); ++child)
                {
                  cell->child(child)->get_mg_dof_indices(dof_indices);
                  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                    dof_values_fine(i) = vectors[level](dof_indices[i]);
                  fe.get_restriction_matrix(child, cell->refinement_case()).vmult (tmp, dof_values_fine);
                  for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                    if (fe.restriction_is_additive(i))
                      dof_values_coarse[i] += tmp[i];
                    else if (tmp(i) != 0.)
                      dof_values_coarse[i] = tmp[i];
                }
              cell->get_mg_dof_indices(dof_indices);
              for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                vectors[level-1](dof_indices[i]) = dof_values_coarse[i];
            }
        // TODO: compress
    }
    
    for(unsigned int level = 0; level <= global_refinements; level++){
      L2Norm<dim, fe_degree_1, value_type> norm(data_1[level]);
      std::cout << level << ": " << norm.run(vectors[level]) << " " << vectors[level].l2_norm() << std::endl;
      
      
      FEEvaluation<dim, fe_degree_1, fe_degree_1+1, 1, value_type> fe_eval(data_1[level], 0);

      for(unsigned int cell = 0; cell < data_1[level].n_macro_cells(); ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(vectors[level], true,false);

        for(unsigned int i = 0; i < fe_eval.static_dofs_per_cell; i++){
            auto point_real = fe_eval.quadrature_point(i)[0];
            auto point_ref  = fe_eval.begin_values()[i];
            
            unsigned int const n_filled_lanes = data_1[level].n_active_entries_per_cell_batch(cell);
            for(unsigned int v = 0; v < n_filled_lanes; v++)
               printf("%10.5f", point_real[v]);
            for(unsigned int v = n_filled_lanes; v < VectorizedArray<value_type>::n_array_elements; v++)
               printf("          ");
            printf("     ");
            for(unsigned int v = 0; v < n_filled_lanes; v++)
               printf("%10.5f", point_ref[v]);
            for(unsigned int v = n_filled_lanes; v < VectorizedArray<value_type>::n_array_elements; v++)
               printf("          ");
            printf("\n");
        }

      }
      
      printf("\n\n");
      
      /*
      MGDataOut<dim> data_out(level);
      data_out.attach_dof_handler(dof_handler);

      data_out.add_data_vector(vectors[global_refinements], "solution");
      data_out.build_patches(fe_degree_1);

      const std::string filename = "output/solution." + std::to_string(level) + ".vtu";
      data_out.write_vtu_in_parallel(filename.c_str(), MPI_COMM_WORLD);
       */
    }

  }
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  int                rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);


      Runner<2, 3> run_cg;
      run_cg.run();
}
