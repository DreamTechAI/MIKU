#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

#include <utility> // 用于 std::pair

#include <numeric>
#include <random>
#include <unordered_set>

#include <cmath>   // std::sqrt, std::ceil
#include <tuple>   // std::tuple


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // 必须包含这个，否则 vector/pair 无法自动转换

namespace py = pybind11;

constexpr float kMaxAspectRatio = 4.0f;
constexpr float kAspectPenaltyWeight = 0.03f;
constexpr float kPortraitPenalty = 0.01f;
constexpr float kUtilizationTieEpsilon = 0.01f;

void enforce_max_aspect_ratio(float& w, float& h) {
    if (w <= 0 || h <= 0) return;

    if (w / h > kMaxAspectRatio) {
        h = std::ceil(w / kMaxAspectRatio);
    } else if (h / w > kMaxAspectRatio) {
        w = std::ceil(h / kMaxAspectRatio);
    }
}

float aspect_ratio(float w, float h) {
    if (w <= 0 || h <= 0) return kMaxAspectRatio;
    return std::max(w / h, h / w);
}

float packing_rank_score(float utilization, float w, float h) {
    float ratio = aspect_ratio(w, h);
    float score = utilization - kAspectPenaltyWeight * (ratio - 1.0f) / (kMaxAspectRatio - 1.0f);
    if (h > w) {
        score -= kPortraitPenalty;
    }
    return score;
}

// 假设之前的 Rect 类定义如下
struct Rect {
    float x, y, w, h;
    std::string rid;
    bool rotated;

    Rect(float x, float y, float w, float h, std::string rid = "", bool rotated = false)
        : x(x), y(y), w(w), h(h), rid(rid), rotated(rotated) {}
};

class MaxRectsBin {
public:
    float width, height;
    std::vector<Rect> used_rects;
    std::vector<Rect> free_rects;
    float total_area = 0;

    MaxRectsBin(float width, float height) : width(width), height(height) {
        // 初始化时，整个箱子是一个大的空闲矩形
        free_rects.push_back(Rect(0, 0, width, height));
    }

    // 1. 计算所有已用矩形的最小包围盒
    void compute_bounding_box() {
        if (used_rects.empty()) return;

        float min_x = std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float max_y = std::numeric_limits<float>::lowest();

        for (const auto& r : used_rects) {
            min_x = std::min(min_x, r.x);
            min_y = std::min(min_y, r.y);
            max_x = std::max(max_x, r.x + r.w);
            max_y = std::max(max_y, r.y + r.h);
        }

        this->width = max_x - min_x;
        this->height = max_y - min_y;
    }

    // 2. 插入新矩形 (BSSF: Best Short Side Fit 策略)
    Rect* insert(float w, float h, std::string rid = "", bool allow_rotate = true) {
        Rect* best_node = nullptr;
        float best_short = std::numeric_limits<float>::max();
        float best_long = std::numeric_limits<float>::max();

        for (const auto& free : free_rects) {
            // 尝试两种方向：不旋转和旋转
            bool options[] = {false, true};
            for (bool rotated : options) {
                if (rotated && !allow_rotate) continue;

                float rw = rotated ? h : w;
                float rh = rotated ? w : h;

                if (rw <= free.w && rh <= free.h) {
                    float leftover_w = std::abs(free.w - rw);
                    float leftover_h = std::abs(free.h - rh);
                    float short_side = std::min(leftover_w, leftover_h);
                    float long_side = std::max(leftover_w, leftover_h);

                    // 选择剩余空间最小（更紧凑）的空闲块
                    if (short_side < best_short || (short_side == best_short && long_side < best_long)) {
                        if (best_node) delete best_node; // 清理上次记录
                        best_node = new Rect(free.x, free.y, rw, rh, rid, rotated);
                        best_short = short_side;
                        best_long = long_side;
                    }
                }
            }
        }

        if (best_node) {
            place_rect(*best_node);
            used_rects.push_back(*best_node);
            Rect* ref = &used_rects.back();
            delete best_node; // 释放临时指针
            return ref;
        }
        return nullptr;
    }

    // 3. 放置矩形并切割空闲区
    void place_rect(const Rect& rect) {
        for (size_t i = 0; i < free_rects.size(); ) {
            if (split_free_rect(free_rects[i], rect)) {
                free_rects.erase(free_rects.begin() + i);
            } else {
                ++i;
            }
        }
        prune_free_list();
    }

    // 4. 切割算法：将一个被占用的空闲矩形切割成多个更小的矩形
    bool split_free_rect(Rect free, const Rect& used) {
        if (!intersect(free, used)) return false;

        // 在垂直方向切割
        if (used.x < free.x + free.w && used.x + used.w > free.x) {
            // 留在上方的空间
            if (used.y > free.y && used.y < free.y + free.h) {
                free_rects.push_back(Rect(free.x, free.y, free.w, used.y - free.y));
            }
            // 留在下方的空间
            if (used.y + used.h < free.y + free.h) {
                free_rects.push_back(Rect(free.x, used.y + used.h, free.w, free.y + free.h - (used.y + used.h)));
            }
        }

        // 在水平方向切割
        if (used.y < free.y + free.h && used.y + used.h > free.y) {
            // 留在左侧的空间
            if (used.x > free.x && used.x < free.x + free.w) {
                free_rects.push_back(Rect(free.x, free.y, used.x - free.x, free.h));
            }
            // 留在右侧的空间
            if (used.x + used.w < free.x + free.w) {
                free_rects.push_back(Rect(used.x + used.w, free.y, free.x + free.w - (used.x + used.w), free.h));
            }
        }
        return true;
    }

    // 5. 剪枝：删除被完全包含在其他矩形内部的空闲矩形
    void prune_free_list() {
        for (size_t i = 0; i < free_rects.size(); ++i) {
            for (size_t j = i + 1; j < free_rects.size(); ) {
                if (is_contained(free_rects[i], free_rects[j])) {
                    free_rects.erase(free_rects.begin() + i);
                    --i;
                    break;
                }
                if (is_contained(free_rects[j], free_rects[i])) {
                    free_rects.erase(free_rects.begin() + j);
                } else {
                    ++j;
                }
            }
        }
    }

    // 工具函数
    static bool intersect(const Rect& a, const Rect& b) {
        return !(b.x >= a.x + a.w || b.x + b.w <= a.x || b.y >= a.y + a.h || b.y + b.h <= a.y);
    }

    static bool is_contained(const Rect& a, const Rect& b) {
        return (a.x >= b.x && a.y >= b.y && a.x + a.w <= b.x + b.w && a.y + a.h <= b.y + b.h);
    }
};

void transpose_bin(MaxRectsBin& bin) {
    for (auto& r : bin.used_rects) {
        std::swap(r.w, r.h);
        std::swap(r.x, r.y);
        r.rotated = !r.rotated;
    }
    std::swap(bin.width, bin.height);
}

std::pair<float, MaxRectsBin> evaluate(
    const std::vector<int>& order, 
    const std::vector<std::pair<float, float>>& rects, 
    float bin_w, 
    float bin_h,
    bool allow_rotate = true
) {
    // 1. 初始化装箱器
    MaxRectsBin bin_pack(bin_w, bin_h);
    float total_area = 0;

    // 2. 按顺序尝试插入矩形
    for (int idx : order) {
        float w = rects[idx].first;
        float h = rects[idx].second;
        
        // 调用之前实现的 insert 方法
        // 传入 std::to_string(idx) 作为 rid
        Rect* node = bin_pack.insert(w, h, std::to_string(idx), allow_rotate);

        if (node) {
            total_area += w * h;
        } else {
            // 如果其中一个放不下，按照原 Python 逻辑返回 0
            // 这里返回一个空的 bin_pack
            return {0.0f, MaxRectsBin(0, 0)};
        }
    }

    // 3. 计算最终边界框并计算利用率
    bin_pack.compute_bounding_box();
    enforce_max_aspect_ratio(bin_pack.width, bin_pack.height);
    
    float bbox_area = bin_pack.width * bin_pack.height;
    float utilization = 0;
    
    if (bbox_area > 0) {
        utilization = total_area / bbox_area;
    }

    // 4. 返回结果对
    return {utilization, bin_pack};
}


std::vector<int> ordered_crossover_part(const std::vector<int>& p1, const std::vector<int>& p2, int cut) {
    std::vector<int> child;
    child.reserve(p1.size());
    std::unordered_set<int> seen;

    // 1. 复制前半段
    for (int i = 0; i < cut; ++i) {
        child.push_back(p1[i]);
        seen.insert(p1[i]);
    }
    // 2. 从父代2中按顺序填充剩余元素（保证唯一性）
    for (int gene : p2) {
        if (seen.find(gene) == seen.end()) {
            child.push_back(gene);
        }
    }
    return child;
}

std::tuple<std::vector<int>, float, MaxRectsBin> genetic_algorithm(
    const std::vector<std::pair<float, float>>& rects,
    float bin_w, float bin_h,
    int pop_size = 40, int generations = 80, float mutation_rate = 0.1,
    bool allow_rotate = true
) {
    int n = rects.size();
    std::vector<std::vector<int>> population;
    
    // 随机数引擎
    std::random_device rd;
    std::mt19937 g(rd());

    // 1. 初始化：第一个个体按面积降序排序 (Heuristic)
    std::vector<int> area_sorted(n);
    std::iota(area_sorted.begin(), area_sorted.end(), 0);
    std::sort(area_sorted.begin(), area_sorted.end(), [&](int i, int j) {
        return (rects[i].first * rects[i].second) > (rects[j].first * rects[j].second);
    });
    population.push_back(area_sorted);

    // 其余个体随机打乱
    for (int i = 1; i < pop_size; ++i) {
        std::vector<int> perm = area_sorted;
        std::shuffle(perm.begin(), perm.end(), g);
        population.push_back(perm);
    }

    std::vector<int> best_solution;
    float best_score = -1.0f;
    MaxRectsBin best_bin(0, 0);

    // 2. 进化循环
    for (int gen = 0; gen < generations; ++gen) {
        std::vector<float> scores;
        for (const auto& individual : population) {
            auto [score, b] = evaluate(individual, rects, bin_w, bin_h, allow_rotate);
            scores.push_back(score);

            if (score > best_score) {
                best_score = score;
                best_solution = individual;
                best_bin = b;
            }
        }

        // 锦标赛选择 (Tournament Selection)
        std::vector<std::vector<int>> new_population;
        std::uniform_int_distribution<int> pop_dist(0, pop_size - 1);
        for (int i = 0; i < pop_size; ++i) {
            int i1 = pop_dist(g);
            int i2 = pop_dist(g);
            new_population.push_back(scores[i1] > scores[i2] ? population[i1] : population[i2]);
        }

        // 交叉 (Crossover)
        std::uniform_real_distribution<float> prob_dist(0.0, 1.0);
        std::uniform_int_distribution<int> cut_dist(1, std::max(1, n - 2));
        for (int i = 0; i < pop_size - 1; i += 2) {
            if (prob_dist(g) < 0.8f) {
                int cut = cut_dist(g);
                auto p1 = new_population[i];
                auto p2 = new_population[i+1];
                new_population[i] = ordered_crossover_part(p1, p2, cut);
                new_population[i+1] = ordered_crossover_part(p2, p1, cut);
            }
        }

        // 变异 (Mutation)
        std::uniform_int_distribution<int> gene_dist(0, n - 1);
        for (auto& ind : new_population) {
            if (prob_dist(g) < mutation_rate) {
                int idx1 = gene_dist(g);
                int idx2 = gene_dist(g);
                std::swap(ind[idx1], ind[idx2]);
            }
        }
        population = std::move(new_population);
    }

    return {best_solution, best_score, best_bin};
}

std::pair<float, float> normalize_initial_bin(
    float area,
    float ratio,
    float min_w,
    float min_h
) {
    float bin_w = std::ceil(std::sqrt(area * ratio));
    float bin_h = std::ceil(area / std::max(1.0f, bin_w));

    bin_w = std::max(bin_w, min_w);
    bin_h = std::max(bin_h, min_h);

    if (bin_w * bin_h < area) {
        bin_h = std::max(bin_h, std::ceil(area / bin_w));
    }
    if (bin_w * bin_h < area) {
        bin_w = std::max(bin_w, std::ceil(area / bin_h));
    }

    enforce_max_aspect_ratio(bin_w, bin_h);

    return {bin_w, bin_h};
}

std::vector<std::pair<float, float>> make_initial_bins(
    const std::vector<std::pair<float, float>>& rects,
    bool allow_rotate
) {
    float sum_area = 0;
    float sum_w = 0;
    float sum_h = 0;
    float max_w = 0;
    float max_h = 0;

    for (const auto& r : rects) {
        sum_area += r.first * r.second;
        sum_w += r.first;
        sum_h += r.second;
        max_w = std::max(max_w, r.first);
        max_h = std::max(max_h, r.second);
    }

    std::vector<std::pair<float, float>> candidates;
    auto add_candidate = [&](float w, float h) {
        w = std::ceil(std::max(w, max_w));
        h = std::ceil(std::max(h, max_h));
        enforce_max_aspect_ratio(w, h);
        for (const auto& c : candidates) {
            if (c.first == w && c.second == h) return;
        }
        candidates.push_back({w, h});
    };

    auto add_ratio = [&](float ratio) {
        if (ratio <= 0) return;
        auto [w, h] = normalize_initial_bin(sum_area, ratio, max_w, max_h);
        add_candidate(w, h);
    };

    add_ratio(1.0f);

    if (!allow_rotate) {
        // Non-rotating packing is sensitive to aspect ratio, so seed both
        // strip-like and area-based bins instead of only starting near square.
        add_candidate(sum_w, max_h);
        add_candidate(max_w, sum_h);

        add_ratio(sum_w / std::max(1.0f, max_h));
        add_ratio(max_w / std::max(1.0f, sum_h));

        float ratios[] = {0.25f, 0.33333334f, 0.5f, 0.75f, 1.3333334f, 2.0f, 3.0f, 4.0f};
        for (float ratio : ratios) {
            add_ratio(ratio);
        }
    }

    return candidates;
}

std::tuple<std::vector<int>, float, MaxRectsBin> search_from_initial_bin(
    const std::vector<std::pair<float, float>>& rects,
    float bin_w,
    float bin_h,
    bool allow_rotate
) {
    int flag = 0;

    while (true) {
        // 运行遗传算法寻找当前尺寸下的最优解
        // 使用 C++17 结构化绑定获取结果
        auto [best_order, best_score, best_bin] = genetic_algorithm(rects, bin_w, bin_h, 40, 80, 0.1, allow_rotate);

        // 如果找到了可行解 (score > 0)
        if (best_score > 0) {
            bool should_transpose = false;
            
            // 检查 ID 为 "0" 的矩形是否被旋转了
            for (const auto& r : best_bin.used_rects) {
                if (r.rid == "0" && r.rotated) {
                    should_transpose = true;
                    break;
                }
            }

            // 4. 执行坐标系转置 (与 Python 逻辑对应)
            if (allow_rotate && should_transpose) {
                transpose_bin(best_bin);
            }

            return {best_order, best_score, best_bin};
        } 
        // 如果当前容器太小放不下，交替增加宽或高
        else {
            if (flag == 1) {
                bin_h += 1;
                flag = 0;
            } else {
                bin_w += 1;
                flag = 1;
            }
        }
    }
}

std::tuple<std::vector<int>, float, MaxRectsBin> pack_main_loop(
    const std::vector<std::pair<float, float>>& rects,
    bool allow_rotate = true
) {
    auto candidates = make_initial_bins(rects, allow_rotate);

    std::vector<int> best_order;
    float best_score = -1.0f;
    float best_rank_score = -std::numeric_limits<float>::max();
    float best_ratio = std::numeric_limits<float>::max();
    bool best_landscape = false;
    MaxRectsBin best_bin(0, 0);

    for (const auto& candidate : candidates) {
        auto [order, score, bin] = search_from_initial_bin(
            rects,
            candidate.first,
            candidate.second,
            allow_rotate
        );

        float best_area = best_bin.width * best_bin.height;
        float area = bin.width * bin.height;
        float rank_score = packing_rank_score(score, bin.width, bin.height);
        float ratio = aspect_ratio(bin.width, bin.height);
        bool landscape = bin.width >= bin.height;
        bool near_best_utilization = score >= best_score - kUtilizationTieEpsilon;
        bool better = false;

        if (rank_score > best_rank_score) {
            better = true;
        } else if (near_best_utilization && ratio == best_ratio && landscape && !best_landscape) {
            better = true;
        } else if (near_best_utilization && ratio < best_ratio) {
            better = true;
        } else if (score == best_score && ratio == best_ratio && area < best_area) {
            better = true;
        }

        if (better) {
            best_order = order;
            best_score = score;
            best_rank_score = rank_score;
            best_ratio = ratio;
            best_landscape = landscape;
            best_bin = bin;
        }
    }

    if (allow_rotate && best_bin.height > best_bin.width) {
        transpose_bin(best_bin);
    }

    return {best_order, best_score, best_bin};
}



// 注意：这里的第一个参数 AutoPack 必须与 setup.py 中的 Extension 名称完全一致
PYBIND11_MODULE(AutoPack, m) {
    m.doc() = "C++ MaxRects Genetic Algorithm Pack Module";

    // 1. 导出 Rect 结构体
    py::class_<Rect>(m, "Rect")
        .def(py::init<float, float, float, float, std::string, bool>(),
             py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), 
             py::arg("rid") = "", py::arg("rotated") = false)
        .def_readwrite("x", &Rect::x)
        .def_readwrite("y", &Rect::y)
        .def_readwrite("w", &Rect::w)
        .def_readwrite("h", &Rect::h)
        .def_readwrite("rid", &Rect::rid)
        .def_readwrite("rotated", &Rect::rotated);

    // 2. 导出 MaxRectsBin 类
    py::class_<MaxRectsBin>(m, "MaxRectsBin")
        .def(py::init<float, float>())
        .def_readwrite("width", &MaxRectsBin::width)
        .def_readwrite("height", &MaxRectsBin::height)
        .def_readwrite("used_rects", &MaxRectsBin::used_rects)
        .def_readwrite("free_rects", &MaxRectsBin::free_rects);

    // 3. 导出主循环函数
    m.def("pack_main_loop", &pack_main_loop, "Run the genetic packing algorithm loop",
          py::arg("rects"), py::arg("allow_rotate") = true);
}
