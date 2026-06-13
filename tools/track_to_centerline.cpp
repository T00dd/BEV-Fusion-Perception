#include "track/build_track_path.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>

// read from csv
static std::vector<Eigen::Vector2d> load_cones_from_csv(const std::string& filename) {
    std::vector<Eigen::Vector2d> cones;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return cones;
    }

    std::string line;
    while (std::getline(file, line)) {
        if(line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ',')) {
            try {
                cones.emplace_back(std::stod(x_str), std::stod(y_str));
            } catch (const std::exception& e) {
                // catch errors related to wrong line/header format
                std::cerr << "Error parsing line: " << line << std::endl;
            }
        }
    }
    return cones;
}

static void save_centerline_to_csv(const std::string& filename, const std::vector<Eigen::Vector2d>& path) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    file << "x,y\n"; // header
    for (const auto& point : path) {
        file << point.x() << "," << point.y() << "\n";
    }
    std::cout << "Centerline saved to: " << filename << std::endl;
}

static void cone_spacing(const std::vector<Eigen::Vector2d>& cones, double& median_out, double& p90_out, double& max_out) {
    if(cones.size() < 2) {
        median_out = p90_out = max_out = 0.0;
        return;
    }
    std::vector<double> steps;
    for(size_t i = 1; i < cones.size(); ++i) {
        steps.push_back((cones[i] - cones[i-1]).norm());
    }
    std::sort(steps.begin(), steps.end());
    median_out = steps[steps.size() / 2];
    p90_out = steps[static_cast<size_t>(steps.size() * 0.9)];
    max_out = steps.back();
}

static void track_width(const std::vector<Eigen::Vector2d>& left_cones, const std::vector<Eigen::Vector2d>& right_cones, double& median_out, double& max_out) {
    std::vector<double> widths;
    for(const auto& left : left_cones) {
        double min_dist = std::numeric_limits<double>::infinity();
        for(const auto& right : right_cones) {
            min_dist = std::min(min_dist, (left - right).norm());
        }
        widths.push_back(min_dist);
    }
    std::sort(widths.begin(), widths.end());
    median_out = widths[widths.size() / 2];
    max_out = widths.back();
}

static track::TrackParams compute_track_params(const std::vector<Eigen::Vector2d>& left_cones, const std::vector<Eigen::Vector2d>& right_cones) {
    double med_l, p90_l, max_l;
    double med_r, p90_r, max_r;
    cone_spacing(left_cones, med_l, p90_l, max_l);
    cone_spacing(right_cones, med_r, p90_r, max_r);

    double median_spacing = (med_l + med_r) * 0.5;
    double max_spacing = std::max(max_l, max_r);

    double med_w, max_w;
    track_width(left_cones, right_cones, med_w, max_w);

    track::TrackParams p;

    p.max_triangle_edge = max_spacing * 1.4;
    p.max_circumradius = max_spacing * 1.3;

    // keep a 40% margin near the medain track width value
    p.min_track_width = std::max(1.5, med_w * 0.6);
    p.max_track_width = med_w * 1.8;

    // the max steps has to cover the largest gap
    p.max_mid_step = max_spacing * 1.3;

    p.k_outlier = 3;
    p.outlier_factor = 2.4;
    p.max_turn_deg = 75.0;

    p.dist_weight = 1.0;
    p.turn_weight = 2.0;
    p.width_weight = 0.8;

    // for sampling consider arounf 1/6 of the median spacing
    p.resample_ds = median_spacing / 6.0;

    std::cout << "Computed track parameters: \n"
              << "  spacing median: " << median_spacing << " max =" << max_spacing << "\n"
              << "  track width median: " << med_w << "\n"
              << "  max_triangle_edge: " << p.max_triangle_edge << "\n"
              << "  max_mid_step: " << p.max_mid_step << "\n"
              << "  min_max_track_width: " << p.min_track_width << " - " << p.max_track_width << "\n"
              << "  resample_ds: " << p.resample_ds << "\n";

    return p;
}

int main(int argc, char** argv) {
    if(argc < 4) {
        std::cerr << "usage: " << argv[0] << " <cones_left.csv> <cones_right.csv> <centerline_output.csv>" << std::endl;
        return 1;
    }

    const std::string left_csv = argv[1];
    const std::string right_csv = argv[2];
    const std::string output_csv = argv[3];

    std::vector<Eigen::Vector2d> left_cones = load_cones_from_csv(left_csv);
    std::vector<Eigen::Vector2d> right_cones = load_cones_from_csv(right_csv);

    if(left_cones.empty() || right_cones.empty()) {
        std::cerr << "Error: No cones loaded. Please check the input files." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << left_cones.size() << " left cones and " << right_cones.size() << " right cones." << std::endl;

    track::TrackParams p = compute_track_params(left_cones, right_cones);
    track::TrackPathResult res = track::build_track_path(left_cones, right_cones, p);

    if(!res.success) {
        std::cerr << "Error: Failed to build track path." << std::endl;
        return 1;
    }

    std::cout << "Reconstruction completed! \n" << "  Midpoints: " << res.ordered_midpoints.size() 
                                                << "\n  Centerline points: " << res.centerline_smooth.size()
                                                << std::endl;

    save_centerline_to_csv(output_csv, res.centerline_smooth);
    return 0;
}