#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_generator.h>


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
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
#include <deal.II/grid/manifold.h>

using namespace dealii;

#include "distance_computer.h"

const int best_of = 10;
const int n_refinements = 5;
typedef double Number;

const MPI_Comm comm = MPI_COMM_WORLD;

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1,
typename number = double>
class LaplaceOperator : public Subscriptor {
public:
    typedef number value_type;
    typedef MatrixFree<dim, number> MF;
    typedef std::pair<unsigned int, unsigned int> Range;
    typedef LaplaceOperator This;

    LaplaceOperator(MatrixFree<dim, number> &data) : data(data) {
        unsigned int n_cells = data.n_cell_batches() + data.n_ghost_cell_batches();
        ip.resize(n_cells);

        for (unsigned int i = 0; i < n_cells; ++i) {
            for (unsigned int v = 0; v < data.n_components_filled(i); ++v) {
                ip[i][v] = i;
            }
        }
    };

    void apply_loop() const {
        int dummy;
        data.loop(&This::local_diagonal_cell,
                &This::local_diagonal_face,
                &This::local_diagonal_boundary, this, dummy, dummy);
    }

private:

    void local_diagonal_cell(const MF &data, int &, const int &,
            const Range &cell_range) const {
        FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

        for (auto cell = cell_range.first; cell < cell_range.second; ++cell) {
            phi.reinit(cell);
            printf("c ");
            auto temp = phi.read_cell_data(ip);
            for (int i = 0; i < 4; i++)
                printf("%3d ", (int) temp[i]);
            printf("\n");
        }
    }

    void local_diagonal_face(const MF &data, int &, const int &,
            const Range &cell_range) const {
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi_m(data, true);
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi_p(data, false);

        for (auto cell = cell_range.first; cell < cell_range.second; ++cell) {
            phi_m.reinit(cell);
            auto temp_m = phi_m.read_cell_data(ip);
            phi_p.reinit(cell);
            auto temp_p = phi_p.read_cell_data(ip);
            printf("f ");
            for (int i = 0; i < 4; i++)
                printf("(%3d-%3d)", (int) temp_m[i], (int) temp_p[i]);
            printf("\n");
        }
    }

    void local_diagonal_boundary(const MF &, int &, const int &, const Range &) const {
    }

    MatrixFree<dim, number> &data;
    AlignedVector<VectorizedArray<number> > ip;
};

template<int dim, int fe_degree>
class Run {
public:

    void
    run(ConvergenceTable& convergence_table) {
        double left = -1, right = +1;
        parallel::distributed::Triangulation<dim> triangulation(comm);

        GridGenerator::hyper_cube(triangulation, left, right);

        for (auto & cell : triangulation)
            for (unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
                    ++face_number) {
                // x-direction
                if ((std::fabs(cell.face(face_number)->center()(0) - left) < 1e-12))
                    cell.face(face_number)->set_all_boundary_ids(0);
                else if ((std::fabs(cell.face(face_number)->center()(0) - right) < 1e-12))
                    cell.face(face_number)->set_all_boundary_ids(1);
                    // y-direction
                else if ((std::fabs(cell.face(face_number)->center()(1) - left) < 1e-12))
                    cell.face(face_number)->set_all_boundary_ids(2);
                else if ((std::fabs(cell.face(face_number)->center()(1) - right) < 1e-12))
                    cell.face(face_number)->set_all_boundary_ids(3);
                    // z-direction
                else if ((std::fabs(cell.face(face_number)->center()(2) - left) < 1e-12))
                    cell.face(face_number)->set_all_boundary_ids(4);
                else if ((std::fabs(cell.face(face_number)->center()(2) - right) < 1e-12))
                    cell.face(face_number)->set_all_boundary_ids(5);
            }

//        std::vector < GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
//                periodic_faces;
//        GridTools::collect_periodic_faces(triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
//        GridTools::collect_periodic_faces(triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
//        GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);
//        triangulation.add_periodicity(periodic_faces);

        triangulation.refine_global(log(std::pow(dim==2 ? 5e7 : 2e7, 1.0 / dim) / (fe_degree + 1)) / log(2));

        DoFHandler<dim> dof_handler(triangulation);
        MappingQGeneric<dim> mapping(1);
        AffineConstraints<double> dummy;
        FE_DGQ<dim> fe(fe_degree);
        dof_handler.distribute_dofs(fe);

        typename MatrixFree<dim, Number>::AdditionalData additional_data;
        additional_data.mapping_update_flags = update_gradients | update_JxW_values;
        additional_data.mapping_update_flags_inner_faces = update_gradients | update_JxW_values | update_values | update_normal_vectors;
        additional_data.mapping_update_flags_boundary_faces =
                update_gradients | update_JxW_values | update_values | update_normal_vectors | update_quadrature_points;
        QGauss<1> quadrature(fe_degree + 1);

        MatrixFree<dim, Number> data;
        data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);

        //        {
        //        LaplaceOperator<dim, fe_degree, fe_degree + 1, Number> op(data);
        //        op.apply_loop();
        //        }
        
        convergence_table.add_value("dim", dim);
        convergence_table.add_value("deg", fe_degree);
        convergence_table.add_value("refs", triangulation.n_global_levels());

        {
            DistanceComputer<dim, fe_degree, fe_degree + 1, Number > op(data);
            op.apply_loop(convergence_table);
        }
    }
};

template<int dim>
void
run() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ConvergenceTable convergence_table;
    {
        Run<dim, 1> runner;
        runner.run(convergence_table);
    }
    {
        Run<dim, 2> runner;
        runner.run(convergence_table);
    }
    {
        Run<dim, 3> runner;
        runner.run(convergence_table);
    }
    {
        Run<dim, 4> runner;
        runner.run(convergence_table);
    }
    {
        Run<dim, 5> runner;
        runner.run(convergence_table);
    }

    if (!rank) {
        std::string file_name = "out." + std::to_string(dim) + ".csv";
        std::ofstream outfile;
        outfile.open(file_name.c_str());
        convergence_table.write_text(std::cout);
        convergence_table.write_text(outfile);
        outfile.close();
    }
}

int
main(int argc, char ** argv) {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    run<2>();
    run<3>();

}