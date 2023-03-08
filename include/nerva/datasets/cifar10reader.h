// Copyright: Wieger Wesselink 2022
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file nerva/datasets/cifar10reader.h
/// \brief add your file description here.

#ifndef NERVA_DATASETS_CIFAR10READER_H
#define NERVA_DATASETS_CIFAR10READER_H

#include "nerva/neural_networks/eigen.h"
#include "nerva/utilities/logger.h"
#include "nerva/utilities/text_utility.h"
#include <filesystem>
#include <random>

namespace nerva {

class cifar10reader
{
  protected:
    eigen::matrix Xtrain;
    eigen::matrix Ttrain;
    eigen::matrix Xvalid;
    eigen::matrix Tvalid;

    static void read_slice(const std::string& filename, eigen::matrix& X, eigen::matrix& T, long start)
    {
      auto to_double = [](std::byte x)
      {
        return static_cast<scalar>(std::to_integer<std::uint8_t>(x));
      };

      auto bytes = read_binary_file(filename);
      if (bytes.size() != 30730000)
      {
        throw std::runtime_error("The size of the file " + filename + " is not equal to 3073000");
      }

      NERVA_LOG(log::verbose) << ".";

      for (long j = 0; j < 10000; j++)
      {
        auto first = bytes.begin() + j * 3073;
        auto class_ = std::to_integer<int>(*first++);
        if (class_ > 9)
        {
          throw std::runtime_error("Invalid class " + std::to_string(class_) + " encountered");
        }
        T(class_, start + j) = 1;
        for (long i = 0; i < 3072; i++)
        {
          // store the data as R1 G1 B1 R2 G2 B2 ...
          auto row = 3 * (i % 1024) + (i / 1024);
          X(row, start + j) = to_double(*first++);
        }
      }
    }

    template <typename Vector>
    std::pair<scalar, scalar> mean_stddev(const Vector& x)
    {
      scalar mu = x.mean();
      scalar sigma = std::sqrt((x.array() - mu).square().sum() / x.size());
      if (std::fabs(sigma) < scalar(1e-10))
      {
        sigma = 1; // do not scale for very small values
      }
      return {mu, sigma};
    }

  public:
    cifar10reader()
        : Xtrain(3072, 50000), Xvalid(3072, 10000)
    {
      Ttrain = eigen::matrix::Zero(10, 50000);
      Tvalid = eigen::matrix::Zero(10, 10000);
    }

    void read(const std::string& directory)
    {
      namespace fs = std::filesystem;
      for (int i = 0; i < 5; i++)
      {
        auto path = fs::path(directory) / fs::path("data_batch_" + std::to_string(i + 1) + ".bin");
        read_slice(path.string(), Xtrain, Ttrain, i * 10000);
      }
      auto path = fs::path(directory) / fs::path("test_batch.bin");
      read_slice(path.string(), Xvalid, Tvalid, 0);
      NERVA_LOG(log::verbose) << std::endl;
    }

    void normalize_data()
    {
      auto normalize = [](eigen::matrix& X)
      {
        X = X.unaryExpr([](scalar t) { return scalar(2) * ((t / scalar(255)) - scalar(0.5)); });
      };

      NERVA_LOG(log::verbose) << "normalizing data" << std::endl;
      normalize(Xtrain);
      normalize(Xvalid);
    }

    std::tuple<eigen::matrix, eigen::matrix, eigen::matrix, eigen::matrix> random_subset(long ntrain, long nvalid, std::mt19937& rng)
    {
      eigen::matrix Xtrain_subset(Xtrain.rows(), ntrain);
      eigen::matrix Ttrain_subset(Ttrain.rows(), ntrain);
      eigen::matrix Xvalid_subset(Xvalid.rows(), nvalid);
      eigen::matrix Tvalid_subset(Tvalid.rows(), nvalid);

      auto make_subset = [&](const eigen::matrix& X, const eigen::matrix& T, eigen::matrix& Xsubset, eigen::matrix& Tsubset)
      {
        std::size_t N = X.cols(); // the number of examples
        std::vector<long> I(N);
        std::iota(I.begin(), I.end(), 0);
        std::shuffle(I.begin(), I.end(), rng);

        for (auto i = 0; i < Xsubset.cols(); i++)
        {
          Xsubset.col(i) = X.col(I[i]);
          Tsubset.col(i) = T.col(I[i]);
        }
      };

      make_subset(Xtrain, Ttrain, Xtrain_subset, Ttrain_subset);
      make_subset(Xvalid, Tvalid, Xvalid_subset, Tvalid_subset);
      NERVA_LOG(log::verbose) << "created random subsets" << std::endl;

      return { Xtrain_subset, Ttrain_subset, Xvalid_subset, Tvalid_subset };
    }

    [[nodiscard]] std::tuple<eigen::matrix, eigen::matrix, eigen::matrix, eigen::matrix> data() const
    {
      return { Xtrain, Ttrain, Xvalid, Tvalid };
    }

    [[nodiscard]] const eigen::matrix& xtrain() const
    {
      return Xtrain;
    }

    [[nodiscard]] const eigen::matrix& ttrain() const
    {
      return Ttrain;
    }

    [[nodiscard]] const eigen::matrix& xvalid() const
    {
      return Xvalid;
    }

    [[nodiscard]] const eigen::matrix& tvalid() const
    {
      return Tvalid;
    }
};

} // namespace nerva

#endif // NERVA_DATASETS_CIFAR10READER_H
