/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2021 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "tools/backendbench.h"

#include <algorithm>

#include "chess/board.h"
#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "search/classic/node.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {
const int kDefaultThreads = 1;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kBatchesId{"batches", "",
                          "Number of batches to run as a benchmark."};
const OptionId kStartBatchSizeId{"start-batch-size", "",
                                 "Start benchmark from this batch size."};
const OptionId kMaxBatchSizeId{"max-batch-size", "",
                               "Maximum batch size to benchmark."};
const OptionId kBatchStepId{"batch-step", "",
                            "Step of batch size in benchmark."};
const OptionId kHeaderOnlyOnceId{"header-only-once", "",
                                 "Print CSV header only once."};
const OptionId kFenId{"fen", "", "Benchmark initial position FEN."};

const OptionId kClippyId{"clippy", "", "Enable helpful assistant."};
const OptionId kSweepDemuxId{"sweep-demux", "",
                             "Sweep demux split configurations."};

struct ProfilePoint {
  int batch;
  double ms;
  double nps;
};

void RunSweepDemux(const OptionsDict& option_dict) {
  const std::string backend_opts =
      option_dict.Get<std::string>(SharedBackendParams::kBackendOptionsId);
  const std::string weights_path =
      option_dict.Get<std::string>(SharedBackendParams::kWeightsId);
  const int max_batch = option_dict.Get<int>(kMaxBatchSizeId);
  const int batch_step = option_dict.Get<int>(kBatchStepId);
  const int batches = option_dict.Get<int>(kBatchesId);

  // Parse sub-backend configurations.
  OptionsDict parsed_opts;
  parsed_opts.AddSubdictFromString(backend_opts);
  auto names = parsed_opts.ListSubdicts();

  if (names.size() < 2) {
    throw Exception("--sweep-demux requires at least 2 sub-backends in -o");
  }

  // Set up position for benchmarking.
  const std::string fen = option_dict.Get<std::string>(kFenId);
  classic::NodeTree tree;
  tree.ResetToPosition(fen, {});
  EvalPosition pos{tree.GetPositionHistory().GetPositions(), {}};

  // Load weights once.
  auto weights = LoadWeights(weights_path);

  // Profile each sub-backend.
  struct BackendProfile {
    std::string name;
    std::string backend_name;
    std::vector<ProfilePoint> points;
  };
  std::vector<BackendProfile> profiles;

  for (const auto& name : names) {
    const auto& sub = parsed_opts.GetSubdict(name);
    // Consume the 'split' key if present (so it won't cause "unknown option").
    sub.GetOrDefault<int>("split", 0);

    std::string backend_name =
        sub.GetOrDefault<std::string>("backend", name);

    auto network =
        NetworkFactory::Get()->Create(backend_name, weights, sub);

    std::cout << "\n=== Profiling sub-backend: " << name
              << " (backend=" << backend_name << ") ===" << std::endl;
    std::cout << "size, mean nps, mean ms,     cv," << std::endl;

    // Encode position once for this sub-backend.
    auto input_format = network->GetCapabilities().input_format;
    int transform;
    auto base_input = EncodePositionForNN(
        input_format, pos.pos, 8, FillEmptyHistory::FEN_ONLY, &transform);

    // Warmup with max batch size.
    {
      auto computation = network->NewComputation();
      for (int k = 0; k < max_batch; k++) {
        InputPlanes copy = base_input;
        computation->AddInput(std::move(copy));
      }
      computation->ComputeBlocking();
    }

    // Profile each batch size.
    std::vector<ProfilePoint> points;
    for (int batch_size = batch_step; batch_size <= max_batch;
         batch_size += batch_step) {
      std::vector<double> durations;
      durations.reserve(batches);
      for (int b = 0; b < batches; b++) {
        auto computation = network->NewComputation();
        for (int k = 0; k < batch_size; k++) {
          InputPlanes copy = base_input;
          computation->AddInput(std::move(copy));
        }
        auto t0 = std::chrono::steady_clock::now();
        computation->ComputeBlocking();
        auto t1 = std::chrono::steady_clock::now();
        durations.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
      }

      double total = 0;
      for (double d : durations) total += d;
      double mean_ms = total / batches;
      double nps = batch_size / mean_ms * 1000.0;

      double var = 0;
      for (double d : durations) {
        double diff = d - mean_ms;
        var += diff * diff;
      }
      double cv = std::sqrt(var / (batches - 1)) / mean_ms;

      // clang-format off
      std::cout << std::setw(4) << batch_size << ","
                << std::fixed << std::setprecision(0)
                << std::setw(9) << nps << ","
                << std::defaultfloat << std::setprecision(4)
                << std::setw(8) << mean_ms << ","
                << std::fixed << std::setprecision(4)
                << std::setw(7) << cv
                << std::endl;
      // clang-format on

      points.push_back({batch_size, mean_ms, nps});
    }

    profiles.push_back({name, backend_name, std::move(points)});
  }

  // Compute predicted throughput for all (g, n) pairs.
  // Currently supports exactly 2 sub-backends.
  if (profiles.size() == 2) {
    struct SweepResult {
      int batch_a;
      int batch_b;
      double predicted_nps;
    };
    std::vector<SweepResult> results;

    const auto& pa = profiles[0].points;
    const auto& pb = profiles[1].points;

    for (const auto& a : pa) {
      for (const auto& b : pb) {
        double bottleneck = std::max(a.ms, b.ms);
        double predicted = (a.batch + b.batch) / bottleneck * 1000.0;
        results.push_back({a.batch, b.batch, predicted});
      }
    }

    // Sort by predicted throughput descending.
    std::sort(results.begin(), results.end(),
              [](const SweepResult& x, const SweepResult& y) {
                return x.predicted_nps > y.predicted_nps;
              });

    std::cout << "\n=== Top 10 Predicted Demux Configurations ===" << std::endl;
    std::cout << std::setw(6) << profiles[0].name << ","
              << std::setw(6) << profiles[1].name << ","
              << std::setw(6) << "total" << ","
              << std::setw(12) << "pred_nps" << std::endl;

    int shown = 0;
    for (const auto& r : results) {
      if (shown >= 10) break;
      // clang-format off
      std::cout << std::setw(6) << r.batch_a << ","
                << std::setw(6) << r.batch_b << ","
                << std::setw(6) << (r.batch_a + r.batch_b) << ","
                << std::fixed << std::setprecision(1)
                << std::setw(12) << r.predicted_nps
                << std::endl;
      // clang-format on
      shown++;
    }

    if (!results.empty()) {
      const auto& best = results[0];
      std::cout << "\nRecommended split: "
                << profiles[0].name << "=" << best.batch_a << ", "
                << profiles[1].name << "=" << best.batch_b
                << " (total=" << (best.batch_a + best.batch_b) << ")"
                << std::endl;
      std::cout << "Add split=<N> to each sub-backend in -o and set "
                << "--minibatch-size="
                << (best.batch_a + best.batch_b)
                << " --max-prefetch=0" << std::endl;
    }
  } else {
    std::cerr << "Sweep analysis currently supports exactly 2 sub-backends."
              << std::endl;
  }
}

void Clippy(std::string title, std::string msg3, std::string best3,
            std::string msg2, std::string best2, std::string msg,
            std::string best) {
  std::cout << "  __" << std::endl;
  std::cout << " /  \\" << std::endl;
  std::cout << " |  |    " << std::string(title.length() + 2, '_') << std::endl;
  std::cout << " +  +   | " << std::string(title.length() + 1, ' ') << "|"
            << std::endl;
  std::cout << "(@)(@) _| " << title << " |" << std::endl;
  std::cout << " |  |  \\  " << std::string(6, ' ') << msg3
            << std::string(4 - best3.length(), ' ') << best3
            << std::string(title.length() - 33, ' ') << "|" << std::endl;
  std::cout << " || |/  | " << std::string(6, ' ') << msg2
            << std::string(4 - best2.length(), ' ') << best2
            << std::string(title.length() - 33, ' ') << "|" << std::endl;
  std::cout << " || ||  | " << std::string(6, ' ') << msg
            << std::string(4 - best.length(), ' ') << best
            << std::string(title.length() - 33, ' ') << "|" << std::endl;
  std::cout << " |\\_/|  |" << std::string(title.length() + 2, '_') << "|"
            << std::endl;
  std::cout << " \\___/" << std::endl;
}
void RunNormalBench(const OptionsDict& option_dict) {
  auto backend = BackendManager::Get()->CreateFromParams(option_dict);
  const int threads = option_dict.Get<int>(kThreadsOptionId);

  classic::NodeTree tree;
  tree.ResetToPosition(option_dict.Get<std::string>(kFenId), {});
  EvalPosition pos{tree.GetPositionHistory().GetPositions(), {}};
  std::vector<std::thread> handles;

  // Do any backend initialization outside the loop.
  auto warm = [&]() {
    // Give GPU enough work to make it go from idle clocks to max clocks.
    for (int i = 0; i < 2; i++) {
      auto warmup = backend->CreateComputation();
      for (int j = 0; j < option_dict.Get<int>(kMaxBatchSizeId); ++j) {
        warmup->AddInput(pos, {});
      }
      warmup->ComputeBlocking();
    }
  };
  for (int t = 1; t < threads; t++) {
    handles.emplace_back(warm);
  }
  warm();
  for (auto& handle : handles) {
    handle.join();
  }
  handles.clear();

  const int batches = option_dict.Get<int>(kBatchesId);

  int best = 1;
  int best2 = 1;
  int best3 = 1;
  float best_nps = 0.0f;
  float best_nps2 = 0.0f;
  float best_nps3 = 0.0f;
  std::optional<std::chrono::time_point<std::chrono::steady_clock>> pending;
  using tp = std::chrono::time_point<std::chrono::steady_clock>;
  std::vector<std::vector<tp>> ends(threads);
  for (auto& vend : ends) {
    vend.resize(batches + 1);
  }
  std::vector<std::chrono::duration<double>> times(batches);
  std::vector<int> thread_counts(threads);
  for (int i = option_dict.Get<int>(kStartBatchSizeId);
       i <= option_dict.Get<int>(kMaxBatchSizeId);
       i += option_dict.Get<int>(kBatchStepId)) {
    handles.reserve(threads);
    std::atomic<int> j{0};

    auto compute = [&](int tid = 0) {
      int count = 0;
      auto& end = ends[tid];
      // Ignore the first batch to let GPU queue fill for stable measurements.
      while (j++ < batches) {
        // Put i copies of tree root node into computation and compute.
        auto computation = backend->CreateComputation();
        for (int k = 0; k < i; k++) {
          computation->AddInput(pos, {});
        }
        computation->ComputeBlocking();
        end[count++] = std::chrono::steady_clock::now();
      }
      thread_counts[tid] = count;
    };

    for (int t = 1; t < threads; t++) {
      handles.emplace_back(compute, t);
    }

    compute(0);
    for (auto& handle : handles) {
      handle.join();
    }

    handles.clear();

    double stddev = 0;
    double total = 0;
    int batches_done = 0;
    for (int t = 0; t < threads; t++) {
      for (int j = 1; j < thread_counts[t]; j++) {
        times[batches_done] = (ends[t][j] - ends[t][j - 1]) / threads;
        total += times[batches_done].count();
        batches_done++;
      }
    }

    double mean = total / batches_done;

    for (int j = 0; j < batches_done; j++) {
      double diff = times[j].count() - mean;
      stddev += diff * diff;
    }
    stddev = std::sqrt(stddev / (batches_done - 1));
    double cv = stddev / mean;

    std::sort(times.begin(), times.begin() + batches_done);

    mean *= 1000;

    const auto nps = i * batches_done / total;
    const auto median = batches_done % 2 == 0
                            ? 2 * i /
                                  (times[batches_done / 2 - 1].count() +
                                   times[batches_done / 2].count())
                            : i / times[batches_done / 2].count();
    if (option_dict.Get<bool>(kHeaderOnlyOnceId)
            ? i == option_dict.Get<int>(kStartBatchSizeId)
            : ((i - option_dict.Get<int>(kStartBatchSizeId)) /
                   option_dict.Get<int>(kBatchStepId) % 32 ==
               0)) {
      std::cout << "size,"
                   " mean nps,"
                   " mean ms,"
                   "   sdev,"
                   "     cv,"
                   " max nps,"
                   "  median,"
                   " min nps,"
                << std::endl;
    }
    // clang-format off
    std::cout << std::setw(4) << i << ","
              << std::fixed << std::setprecision(0)
              << std::setw(9) << nps << ","
              << std::defaultfloat << std::setprecision(4)
              << std::setw(8) << mean  << ","
              << std::fixed << std::setprecision(4)
              << std::setw(7) << stddev * 1000 << ","
              << std::setw(7) << cv << ","
              << std::fixed << std::setprecision(0)
              << std::setw(8) << i / times[0].count() << ","
              << std::setw(8) << median << ","
              << std::setw(8) << i / times[batches_done - 1].count()
              << std::endl;
    // clang-format on

    if (option_dict.Get<bool>(kClippyId)) {
      float nps_ingame = std::pow((nps + best_nps) / 2, 1.085);
      float nps_ingame2 = std::pow((nps + best_nps2) / 2, 1.085);
      float nps_ingame3 = std::pow((nps + best_nps3) / 2, 1.085);
      float threshold = 0.16947 * exp(-4.1695e-6 * nps_ingame * 180) + 0.02;
      float threshold2 = 0.16947 * exp(-4.1695e-6 * nps_ingame2 * 15) + 0.02;
      float threshold3 = 0.16947 * exp(-4.1695e-6 * nps_ingame3 * 1) + 0.02;

      if (nps > best_nps &&
          threshold * (i - best) * best_nps < (nps - best_nps) * best) {
        best_nps = nps;
        best = i;
        if (threshold2 * (i - best2) * best_nps2 <
            (nps - best_nps2) * best2) {
          best_nps2 = nps;
          best2 = i;
          if (threshold3 * (i - best3) * best_nps3 <
              (nps - best_nps3) * best3) {
            best_nps3 = nps;
            best3 = i;
          }
        }
        if (!pending) {
          pending = std::chrono::steady_clock::now();
        }
      }
      if (pending) {
        std::chrono::duration<double> time =
            std::chrono::steady_clock::now() - *pending;
        if (time.count() > 10) {
          Clippy("Recommended minibatch-size for this net (so far):",
                 "1s/move   (Bullet):     ", std::to_string(best3),
                 "15s/move  (Rapid):      ", std::to_string(best2),
                 "3min/move (Tournament): ", std::to_string(best));
          pending.reset();
        }
      }
    }
  }
  if (option_dict.Get<bool>(kClippyId)) {
    Clippy("Recommended minibatch-size for this net:",
           "1s/move   (Bullet):     ", std::to_string(best3),
           "15s/move  (Rapid):      ", std::to_string(best2),
           "3min/move (Tournament): ", std::to_string(best));
  }
}

}  // namespace

void BackendBenchmark::Run() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;

  options.Add<IntOption>(kBatchesId, 1, 999999999) = 100;
  options.Add<IntOption>(kStartBatchSizeId, 1, 1024) = 1;
  options.Add<IntOption>(kMaxBatchSizeId, 1, 1024) = 256;
  options.Add<IntOption>(kBatchStepId, 1, 256) = 1;
  options.Add<BoolOption>(kHeaderOnlyOnceId) = false;
  options.Add<StringOption>(kFenId) = ChessBoard::kStartposFen;
  options.Add<BoolOption>(kClippyId) = false;
  options.Add<BoolOption>(kSweepDemuxId) = false;

  if (!options.ProcessAllFlags()) return;

  try {
    auto option_dict = options.GetOptionsDict();

    if (option_dict.Get<bool>(kSweepDemuxId)) {
      RunSweepDemux(option_dict);
    } else {
      RunNormalBench(option_dict);
    }
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}
}  // namespace lczero
