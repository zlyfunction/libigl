#include <igl/MappingEnergyType.h>
#include <igl/PI.h>
#include <igl/Timer.h>
#include <igl/arap.h>
#include <igl/boundary_loop.h>
#include <igl/cat.h>
#include <igl/doublearea.h>
#include <igl/flipped_triangles.h>
#include <igl/harmonic.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOBJ.h>
#include <igl/topological_hole_fill.h>
#include <igl/triangle/scaf.h>

// Define matrix here
Eigen::MatrixXd V_after, V_before, V_joint_before, V_joint_after;
Eigen::MatrixXi F_after, F_before, F_joint, F_joint_before, F_joint_after;

bool is_bd_v0 = false;
bool is_bd_v1 = false;

// Soft Constraints
Eigen::VectorXi b_soft;
Eigen::MatrixXd bc_soft;
// Hard Constraints
std::vector<std::pair<int, int>> b_hard;

igl::triangle::SCAFData scaf_data;

igl::Timer timer;

int show_option = 1;

std::vector<int> v_id_map_before, v_id_map_after;

// read mesh function
void read_mesh(const std::string &mesh_path, int operation_number) {
  igl::readOBJ(mesh_path + "VF_before_" + std::to_string(operation_number) +
                   ".obj",
               V_before, F_before);
  igl::readOBJ(mesh_path + "VF_after_" + std::to_string(operation_number) +
                   ".obj",
               V_after, F_after);
  // read other information
  std::ifstream v_id_map_file(mesh_path + "v_id_map_" +
                              std::to_string(operation_number) + ".txt");
  std::string line;
  // read "boundary case:" line
  std::getline(v_id_map_file, line);

  // read "is_bd_v0" and "is_bd_v1"
  std::getline(v_id_map_file, line);
  std::istringstream bd_iss(line);
  bd_iss >> is_bd_v0 >> is_bd_v1;

  // read "v_id_map_before:"
  std::getline(v_id_map_file, line); // skip "v_id_map_before:" line
  std::getline(v_id_map_file, line);
  std::istringstream before_iss(line);
  int id;
  while (before_iss >> id) {
    v_id_map_before.push_back(id);
  }

  // read "v_id_map_after:"
  std::getline(v_id_map_file, line); // skip "v_id_map_after:" line
  std::getline(v_id_map_file, line);
  std::istringstream after_iss(line);
  while (after_iss >> id) {
    v_id_map_after.push_back(id);
  }

  // close file
  v_id_map_file.close();

  // output debug information
  {
    std::cout << "Boundary case:\n";
    std::cout << "is_bd_v0: " << is_bd_v0 << ", is_bd_v1: " << is_bd_v1 << "\n";

    std::cout << "v_id_map_before: \n";
    for (int val : v_id_map_before) {
      std::cout << val << " ";
    }

    std::cout << "\nv_id_map_after: \n";
    for (int val : v_id_map_after) {
      std::cout << val << " ";
    }
    std::cout << std::endl;
  }
}

void get_local_vid_map(std::vector<int> &local_vid_after_to_before_map,
                       int &vi_before, int &vj_before, int &vi_after) {
  vi_after = 0;

  local_vid_after_to_before_map.resize(v_id_map_after.size(), -1);
  for (int i = 1; i < v_id_map_after.size(); i++) {
    auto it = std::find(v_id_map_before.begin(), v_id_map_before.end(),
                        v_id_map_after[i]);
    if (it == v_id_map_before.end()) {
      std::runtime_error("Error: vertex not found in v_id_map_before");
    }
    local_vid_after_to_before_map[i] =
        std::distance(v_id_map_before.begin(), it);
  }
  vi_before = 0;
  vj_before = -1;
  for (int i = 0; i < v_id_map_before.size(); i++) {
    if (std::find(v_id_map_after.begin(), v_id_map_after.end(),
                  v_id_map_before[i]) == v_id_map_after.end()) {
      vj_before = i;
      break;
    }
  }

  std::cout << "vj_before = " << vj_before << std::endl;
  if (vj_before == -1 || vj_before == vi_before) {
    throw std::runtime_error("Cannot find the joint vertex!");
  }
}

void get_joint_mesh(int case_id) {
  // input is_bd_v0, is_bd_v1
  // case 0, interior case
  int vi_before, vj_before, vi_after;

  if (case_id == 0) {
    std::vector<int> local_vid_after_to_before_map(v_id_map_after.size(), -1);

    get_local_vid_map(local_vid_after_to_before_map, vi_before, vj_before,
                      vi_after);

    // for collapse operation there is one more vertex in the after mesh
    int N_v_joint = V_before.rows() + 1;

    // build V_joint_before, and V_joint_after
    V_joint_before = V_before;
    V_joint_before.conservativeResize(N_v_joint, V_before.cols());
    V_joint_before.row(V_before.rows()) = V_after.row(vi_after);

    V_joint_after = V_joint_before;

    // joint the two meshes
    // get F_joint,(first after, then before)
    F_joint_before = F_before;
    F_joint_after.resize(F_after.rows(), F_after.cols());
    // build F_joint_after
    {
      local_vid_after_to_before_map[vi_after] = N_v_joint - 1;
      for (int i = 0; i < F_joint_after.rows(); i++) {
        for (int j = 0; j < F_joint_after.cols(); j++) {
          F_joint_after(i, j) = local_vid_after_to_before_map[F_after(i, j)];
        }
      }
    }

    // build F_joint = [F_joint_before; F_joint_after]
    F_joint.conservativeResize(F_joint_after.rows() + F_joint_before.rows(),
                               F_joint_after.cols());
    F_joint.topRows(F_joint_before.rows()) = F_joint_before;
    F_joint.bottomRows(F_joint_after.rows()) = F_joint_after;

  } // case 0
  else if (case_id == 1) {
    std::vector<int> local_vid_after_to_before_map(v_id_map_after.size(), -1);

    get_local_vid_map(local_vid_after_to_before_map, vi_before, vj_before,
                      vi_after);

    // for connector case there will be no more vertices than before case
    int N_v_joint = V_before.rows();

    // build V_joint_before, and V_joint_after
    V_joint_before = V_before;
    V_joint_after = V_joint_before;
    // which vertex is on the boundary
    // note that vj is the one to be kept
    int v_bd = is_bd_v0 ? vj_before : vi_before;
    V_joint_after.row(v_bd) = V_after.row(vi_after);

    // joint the two meshes
    // get F_joint,(first after, then before)
    F_joint_before = F_before;
    F_joint_after.resize(F_after.rows(), F_after.cols());
    // build F_joint_after
    {
      local_vid_after_to_before_map[vi_after] =
          v_bd; // Note this line is different from case 0
      for (int i = 0; i < F_joint_after.rows(); i++) {
        for (int j = 0; j < F_joint_after.cols(); j++) {
          F_joint_after(i, j) = local_vid_after_to_before_map[F_after(i, j)];
        }
      }
    }

    // build F_joint = [F_joint_before; F_joint_after]
    F_joint.conservativeResize(F_joint_after.rows() + F_joint_before.rows(),
                               F_joint_after.cols());
    F_joint.topRows(F_joint_before.rows()) = F_joint_before;
    F_joint.bottomRows(F_joint_after.rows()) = F_joint_after;

  } // case 1
  else if (case_id == 2) {

    bool do_3_colinear_case = true;
    std::vector<int> local_vid_after_to_before_map(v_id_map_after.size(), -1);

    get_local_vid_map(local_vid_after_to_before_map, vi_before, vj_before,
                      vi_after);

    // we need to decide which vertex to keep
    int v_to_keep;
    {
      Eigen::VectorXi bd_loop_before;
      igl::boundary_loop(F_before, bd_loop_before);

      int i_idx = std::distance(
          bd_loop_before.begin(),
          std::find(bd_loop_before.begin(), bd_loop_before.end(), vi_before));
      int j_idx = std::distance(
          bd_loop_before.begin(),
          std::find(bd_loop_before.begin(), bd_loop_before.end(), vj_before));
      int offset = 1;
      if (bd_loop_before[i_idx + offset] == vj_before) {
        offset = -1;
      }
      // DEBUG CHECK
      if (bd_loop_before[(i_idx + bd_loop_before.size() - offset) %
                         bd_loop_before.size()] != vj_before) {
        std::runtime_error(
            "Something wrong with the boundary loop in 3-colinear method");
      }
      // vp, vi, vj, vq in order
      int v_p = bd_loop_before[(i_idx + bd_loop_before.size() + offset) %
                               bd_loop_before.size()];
      int v_q = bd_loop_before[(j_idx + bd_loop_before.size() - offset) %
                               bd_loop_before.size()];

      auto in_same_triangle = [&](int v0, int v2, int v3) {
        for (int i = 0; i < F_before.rows(); i++) {
          if ((F_before(i, 0) == v0 || F_before(i, 1) == v0 ||
               F_before(i, 2) == v0) &&
              (F_before(i, 0) == v2 || F_before(i, 1) == v2 ||
               F_before(i, 2) == v2) &&
              (F_before(i, 0) == v3 || F_before(i, 1) == v3 ||
               F_before(i, 2) == v3)) {
            return true;
          }
        }
        return false;
      };

      b_soft.resize(3);

      if (in_same_triangle(vi_before, vj_before, v_q)) {
        v_to_keep = vj_before;
        // keep vj means vp, vi, vj colinear
        b_soft << v_p, vi_before, vj_before;
        b_hard.resize(3);
        b_hard[0] = std::make_pair(v_p, 0);
        b_hard[1] = std::make_pair(vi_before, 0);
        b_hard[2] = std::make_pair(vj_before, 0);
      } else {
        v_to_keep = vi_before;
        // keep vi means vp, vi, vq colinear
        b_soft << vi_before, vj_before, v_q;
        b_hard.resize(3);
        b_hard[0] = std::make_pair(vi_before, 0);
        b_hard[1] = std::make_pair(vj_before, 0);
        b_hard[2] = std::make_pair(v_q, 0);
      }
    }

    // TODO: implement this
    // TODO: 3 colinear method or 5 colinear method?
    if (do_3_colinear_case) {

      // for connector case there will be no more vertices than before case
      int N_v_joint = V_before.rows();

      // build V_joint_before, and V_joint_after
      V_joint_before = V_before;
      V_joint_after = V_joint_before;
      V_joint_after.row(v_to_keep) = V_after.row(vi_after);

      // joint the two meshes
      // get F_joint,(first after, then before)
      F_joint_before = F_before;
      F_joint_after.resize(F_after.rows(), F_after.cols());
      // build F_joint_after
      {
        local_vid_after_to_before_map[vi_after] =
            v_to_keep; // Note this line is different from case 0
        for (int i = 0; i < F_joint_after.rows(); i++) {
          for (int j = 0; j < F_joint_after.cols(); j++) {
            F_joint_after(i, j) = local_vid_after_to_before_map[F_after(i, j)];
          }
        }
      }

      // build F_joint = [F_joint_before; F_joint_after]
      F_joint.conservativeResize(F_joint_after.rows() + F_joint_before.rows(),
                                 F_joint_after.cols());
      F_joint.topRows(F_joint_before.rows()) = F_joint_before;
      F_joint.bottomRows(F_joint_after.rows()) = F_joint_after;

    } else {
      int N_v_joint = V_before.rows() + 1;

      // build V_joint_before, and V_joint_after
      V_joint_before = V_before;
      V_joint_before.conservativeResize(N_v_joint, V_before.cols());
      V_joint_before.row(V_before.rows()) = V_after.row(vi_after);
      V_joint_after = V_joint_before;

      // joint the two meshes
      // get F_joint,(first after, then before)
      F_joint_before = F_before;
      F_joint_after.resize(F_after.rows(), F_after.cols());
      // build F_joint_after
      {
        local_vid_after_to_before_map[vi_after] = N_v_joint - 1;
        for (int i = 0; i < F_joint_after.rows(); i++) {
          for (int j = 0; j < F_joint_after.cols(); j++) {
            F_joint_after(i, j) = local_vid_after_to_before_map[F_after(i, j)];
          }
        }
      }
      std::runtime_error("4-Colinear-Method is Not implemented yet");
    }
  } // case 2
}

int main(int argc, char *argv[]) {
  using namespace std;

  // Load before and after mesh
  std::string mesh_folder_path = "/Users/leyi/Developer/wmtk_main/build/"
                                 "operation_log_test_2d_circle/";
  //  "operation_log_test1/";
  int operation_id = 10;

  if (argc > 1) {
    operation_id = std::stoi(argv[1]);
  }
  read_mesh(mesh_folder_path, operation_id);

  // get case id, 0 for interior case, 2 for boundary case, 1 for connector
  // case
  int case_id = 0;
  if (is_bd_v0)
    case_id += 1;
  if (is_bd_v1)
    case_id += 1;
  std::cout << "Boundary case id: " << case_id << std::endl;

  // TODO: get joint mesh here
  get_joint_mesh(case_id);

  // debug display 0
  {
    auto key_down_init = [&](igl::opengl::glfw::Viewer &viewer,
                             unsigned char key, int modifier) {
      if (key == '1')
        show_option = 1;
      else if (key == '2')
        show_option = 2;
      else if (key == '3')
        show_option = 3;

      switch (show_option) {
      case 1:
        viewer.data().clear();
        viewer.data().set_mesh(V_joint_before, F_joint_before);
        viewer.core().align_camera_center(V_joint_before, F_joint_before);
        break;
      case 2:
        viewer.data().clear();
        viewer.data().set_mesh(V_joint_after, F_joint_after);
        viewer.core().align_camera_center(V_joint_before, F_joint_before);
        break;
      case 3:
        viewer.data().clear();
        viewer.data().set_mesh(V_joint_before, F_joint_after);
        viewer.core().align_camera_center(V_joint_before, F_joint_before);
        break;
      }

      return false;
    };
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V_joint_before, F_joint_before);
    viewer.callback_key_down = key_down_init;
    viewer.launch();
  }

  // get uv_init
  Eigen::MatrixXd bnd_uv, uv_init;
  Eigen::VectorXi bnd;

  // boundary loop is computed by F_before, since F_joint is theorectically
  // closed
  igl::boundary_loop(F_before, bnd);

  std::cout << "Boundary Size: " << bnd.size() << std::endl;
  std::cout << "Boundary loop: " << bnd.transpose() << std::endl;
  Eigen::MatrixXd M_before;
  igl::doublearea(V_joint_before, F_joint_before, M_before);

  igl::map_vertices_to_circle(V_joint_before, bnd, bnd_uv);
  bnd_uv *= sqrt(M_before.sum() / (2 * igl::PI));

  if (case_id == 2) {
    std::cout << "Case 2: force it to be colinear\n";
    bc_soft.resize(b_soft.size(), 2);
    double fixed_value = 0;
    bool find = false;
    for (int i = 0; i < bnd.size(); i++) {
      for (int j = 0; j < b_soft.size(); j++)
        if (bnd[i] == b_soft[j]) {
          if (find) {
            bnd_uv(i, 0) = fixed_value;
            bc_soft.row(j) = bnd_uv.row(i);
          } else {
            find = true;
            fixed_value = bnd_uv(i, 0);
            bc_soft.row(j) = bnd_uv.row(i);
          }
        }
    }
  }

  // get uv_init
  igl::harmonic(V_joint_before, F_joint, bnd, bnd_uv, 1, uv_init);

  // debug display 1
  /*
  {
    auto key_down_debug = [&](igl::opengl::glfw::Viewer &viewer,
                              unsigned char key, int modifier) {
      if (key == '1')
        show_option = 1;
      else if (key == '2')
        show_option = 2;
      else if (key == '3')
        show_option = 3;

      switch (show_option) {
      case 1:
        viewer.data().clear();
        viewer.data().set_mesh(uv_init, F_joint_before);
        viewer.core().align_camera_center(uv_init, F_joint_before);
        break;
      case 2:
        viewer.data().clear();
        viewer.data().set_mesh(uv_init, F_joint_after);
        viewer.core().align_camera_center(uv_init, F_joint_before);
        break;
      case 3:
        viewer.data().clear();
        viewer.data().set_mesh(uv_init, F_joint);
        viewer.core().align_camera_center(uv_init, F_joint_before);
        break;
      }

      return false;
    };

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(uv_init, F_joint_before);
    viewer.callback_key_down = key_down_debug;
    viewer.launch();
  }
  */

  // TODO: need modification here on this? for hard constraint
  igl::triangle::scaf_precompute_joint(
      V_joint_before, V_joint_after, F_joint, F_joint_before, F_joint_after,
      uv_init, scaf_data, igl::MappingEnergyType::SYMMETRIC_DIRICHLET, b_soft,
      bc_soft, 0, b_hard);

  // debug display 2
  {
    auto key_down_debug = [&](igl::opengl::glfw::Viewer &viewer,
                              unsigned char key, int modifier) {
      if (key == '1')
        show_option = 1;
      else if (key == '2')
        show_option = 2;
      else if (key == '3')
        show_option = 3;
      else if (key == '4')
        show_option = 4;

      Eigen::MatrixXi F_all;
      Eigen::MatrixXd shifted_uv;
      shifted_uv = scaf_data.w_uv;
      double max_x_axis = shifted_uv.col(0).maxCoeff();
      double min_x_axis = shifted_uv.col(0).minCoeff();
      shifted_uv.col(0).array() += 1.1 * (max_x_axis - min_x_axis);
      Eigen::MatrixXd uv_all(scaf_data.w_uv.rows() * 2, scaf_data.w_uv.cols());
      uv_all << scaf_data.w_uv, shifted_uv;

      if (key == ' ') {
        timer.start();
        igl::triangle::scaf_solve(scaf_data, 1);
        std::cout << "time = " << timer.getElapsedTime() << std::endl;
        shifted_uv = scaf_data.w_uv;
        max_x_axis = shifted_uv.col(0).maxCoeff();
        min_x_axis = shifted_uv.col(0).minCoeff();
        shifted_uv.col(0).array() += 1.1 * (max_x_axis - min_x_axis);
        uv_all.resize(scaf_data.w_uv.rows() * 2, scaf_data.w_uv.cols());
        uv_all << scaf_data.w_uv, shifted_uv;
      }

      // get max y coordinate of scaf_data.w_uv

      switch (show_option) {
      case 1: {
        viewer.data().clear();
        F_all.resize(F_joint_before.rows() + F_joint_after.rows(),
                     F_joint_before.cols());
        F_all << F_joint_before,
            (F_joint_after.array() + scaf_data.w_uv.rows());

        viewer.data().set_mesh(uv_all, F_all);
        viewer.data().set_colors(Eigen::RowVector3d(230, 220, 170) / 255.0);
        viewer.core().align_camera_center(uv_init, F_joint_before);
        break;
      }
      case 2: {
        viewer.data().clear();
        Eigen::MatrixXi F_w_before, F_w_after;
        Eigen::MatrixXd C;
        igl::cat(1, F_joint_before, scaf_data.s_T, F_w_before);
        igl::cat(1, F_joint_after, scaf_data.s_T, F_w_after);
        C.resize(F_w_before.rows() + F_w_after.rows(), 3);
        for (int i = 0; i < F_joint_before.rows(); i++) {
          C.row(i) = Eigen::RowVector3d(230, 220, 170) / 255.0;
        }
        for (int i = 0; i < scaf_data.s_T.rows(); i++) {
          C.row(i + F_joint_before.rows()) =
              Eigen::RowVector3d(210, 150, 150) / 255.0;
        }
        for (int i = 0; i < F_joint_after.rows(); i++) {
          C.row(i + F_w_before.rows()) =
              Eigen::RowVector3d(230, 220, 170) / 255.0;
        }
        for (int i = 0; i < scaf_data.s_T.rows(); i++) {
          C.row(i + F_w_before.rows() + F_joint_after.rows()) =
              Eigen::RowVector3d(210, 150, 150) / 255.0;
        }
        F_all.resize(F_w_before.rows() + F_w_after.rows(), F_w_before.cols());
        F_all << F_w_before, (F_w_after.array() + scaf_data.w_uv.rows());
        viewer.data().set_mesh(uv_all, F_all);
        viewer.data().set_colors(C);
        viewer.core().align_camera_center(uv_init, F_joint_before);

        break;
      }
      case 3: {
        viewer.data().clear();
        viewer.data().set_mesh(scaf_data.w_uv, F_joint_before);
        viewer.data().set_colors(Eigen::RowVector3d(230, 220, 170) / 255.0);
        viewer.core().align_camera_center(uv_init, F_joint_before);

        break;
      }
      case 4: {
        viewer.data().clear();
        viewer.data().set_mesh(scaf_data.w_uv, F_joint_after);
        viewer.data().set_colors(Eigen::RowVector3d(230, 220, 170) / 255.0);
        viewer.core().align_camera_center(uv_init, F_joint_before);
        break;
      }
      }

      return false;
    };

    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(scaf_data.w_uv, F_joint_before);
    viewer.data().set_colors(Eigen::RowVector3d(230, 220, 170) / 255.0);
    viewer.callback_key_down = key_down_debug;
    viewer.launch();
  }

  return 0;
}
