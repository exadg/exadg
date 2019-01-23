#ifndef DISTANCE_COMPUTER
#define DISTANCE_COMPUTER

struct Interval{
    int begin;
    int end;
    int color = -1;
};

template <int dim, int fe_degree, int n_q_points_1d = fe_degree + 1,
typename number = double>
class DistanceComputer : public Subscriptor {
public:
    typedef number value_type;
    typedef MatrixFree<dim, number> MF;
    typedef std::pair<unsigned int, unsigned int> Range;
    typedef DistanceComputer This;

    DistanceComputer(MatrixFree<dim, number> &data) : data(data), counter(0) {
        this->n_cell_batches_full = data.n_cell_batches() * 4;
        unsigned int n_cells = data.n_cell_batches() + data.n_ghost_cell_batches();
        ip.resize(n_cells);

//        std::cout << n_cells*4 << " " << data.n_cell_batches() * 4 << " " << data.n_ghost_cell_batches()*4 << std::endl;
        
        for (unsigned int i = 0; i < n_cells; ++i) {
            for (unsigned int v = 0; v < data.n_components_filled(i); ++v) {
                ip[i][v] = i*4+v;
            }
        }
        
        visits.resize(n_cells*4);
    };

    void apply_loop(ConvergenceTable& convergence_table) const {
        int dummy;
        data.loop(&This::local_diagonal_cell,
                &This::local_diagonal_face,
                &This::local_diagonal_boundary, this, dummy, dummy);
        
        int bb = 0;
        int cc = 0;
        
        for(auto & set : visits){
            std::set<int> temp;
            int b = * set.begin();
            for(auto element : set){
                if(element-b < 20){
                    bb++;
                    temp.insert(element);
                }
                else
                    cc++;
            }
            
            set = temp;
        }
//        for(auto & set : visits){
//            if(set.size()<=1)
//                continue;
//            printf("%4d: ", *set.rbegin()-*set.begin());
//            for(auto element : set)
//                printf("%4d ", element);
//            printf("\n");
//        }
//        
//        std::cout << bb << " " << cc << std::endl;
        
        std::vector<Interval> intervals;
        
        double n_visits = 0;
        double w_visits = 0;
        for(auto & set : visits){
            
            if(set.size() <= 1)
                continue;
            
            int b = *set.begin();
            int e = *set.rbegin();
            
            AssertThrow(e-b>0, ExcMessage("Error!"));
            
            n_visits++;
            w_visits+=(e-b);
            
            Interval temp;
            temp.begin = b;
            temp.end   = e;
            intervals.push_back(temp);
        }
        
        int m = 0;
        for (auto i : intervals)
            m = std::max(m, i.end);
        
        std::vector<std::set<int>> init(m+1);
        std::vector<std::set<int>> exit(m+1);
        for (unsigned int i = 0; i < intervals.size(); i++){
            auto interval = intervals[i];
            init[interval.begin].insert(i);
            exit[interval.end].insert(i);
        }
        
        std::stack<int> stack;
        for(int i = 0; i < 100000; i++)
            stack.push(i);
        
        for (int i = 0; i <= m; i++){
            for(auto ii : init[i]){
                intervals[ii].color = stack.top();
                stack.pop();
            }
            for(auto& ii : exit[i]){
                stack.push(intervals[ii].color);
            }
        }
        
        
        std::set<int> used_colors;
        for(auto i : intervals){
            if(i.color == -1)
                std::cout << "error" << std::endl;
            used_colors.insert(i.color);
        }
        
        
        convergence_table.add_value("intervals", (int) intervals.size());
        convergence_table.add_value("n_colors", (int) used_colors.size());
        convergence_table.add_value("length", (int) (w_visits/n_visits)*4);
        
        
    }

private:

    void local_diagonal_cell(const MF &data, int &, const int &,
            const Range &cell_range) const {
        FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

        for (auto cell = cell_range.first; cell < cell_range.second; ++cell) {
            phi.reinit(cell);
            auto temp = phi.read_cell_data(ip);
//            printf("c ");
//            for (unsigned int i = 0; i < data.n_active_entries_per_cell_batch(cell); i++)
//                printf("%3d ", (int) temp[i]);
//            printf("\n");
            for (unsigned int i = 0; i < data.n_active_entries_per_cell_batch(cell); i++){
                if(!(0 <= temp[i] && temp[i] < visits.size()))
                    AssertThrow(false, ExcMessage("Error!"));
                visits[(int)temp[i]].insert(counter);
            }
            
            counter++;
        }
    }

    void local_diagonal_face(const MF &data, int &, const int &,
            const Range &cell_range) const {
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi_m(data,true);
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi_p(data,false);

        for (auto cell = cell_range.first; cell < cell_range.second; ++cell) {
            phi_m.reinit(cell);
            auto temp_m = phi_m.read_cell_data(ip);
            phi_p.reinit(cell);
            auto temp_p = phi_p.read_cell_data(ip);
            printf("f ");
            for (unsigned int i = 0; i < data.n_active_entries_per_face_batch(cell); i++)
                printf("(%3d-%3d)", (int) temp_m[i], (int) temp_p[i]);
            printf("\n");
            for (unsigned int i = 0; i < data.n_active_entries_per_face_batch(cell); i++){
                if(!((0 <= temp_m[i] && temp_m[i] < visits.size()) || (0 <= temp_p[i] && temp_p[i] < visits.size())))
                    AssertThrow(false, ExcMessage("Error!"));
//                if(temp_m[i] >= n_cell_batches_full || temp_p[i] >= n_cell_batches_full)
//                    continue;
                visits[(int) temp_m[i]].insert(counter);
                visits[(int) temp_p[i]].insert(counter);
            }
        }
    }

    void local_diagonal_boundary(const MF &, int &, const int &, const Range &) const {
    }

    MatrixFree<dim, number> &data;
    mutable AlignedVector<VectorizedArray<number> > ip;
    mutable std::vector<std::set<int> > visits;
    mutable int counter;
    mutable int n_cell_batches_full;
};

#endif