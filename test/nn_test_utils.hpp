#pragma once

#include <fstream>
#include "matrix.hpp"
#include <iostream>

namespace libai::test_util
{
	template<typename T, size_t input_count>
	struct normalizer
	{
		matrix<T, input_count>  min_;
		matrix<T, input_count>  max_;

		normalizer()
		{
			for (auto& m : min_) m = std::numeric_limits<T>::max();
			for (auto& m : max_) m = std::numeric_limits<T>::min();
		}

		void fit(const matrix<T, input_count>& inputs)
		{
			const auto& f = inputs.flatten();
			for (auto i = 0U; i < f.size(); i++)
			{
				min_[i] = std::min(min_[i], f[i]);
				max_[i] = std::max(max_[i], f[i]);
			}
		}

		void normalize(matrix<T, input_count>& inputs)
		{
			auto& f = inputs.flatten();
			for (auto i = 0U; i < f.size(); i++)
			{
				f[i] = (f[i] - min_[i]) / (max_[i] - min_[i]);
			}
		}
	};

	template<typename T, size_t N>
	class csv_file
	{
		normalizer<T, N> normalizer_;
		bool hack_ = false;
		std::ifstream file_;
		matrix<T, N> data_;

		bool forward()
		{
			if (file_.eof())
			{
				data_ = {};
				return true;
			}

			std::string buffer;
			for (auto i = 0U; i < N; i++)
			{
				getline(file_, buffer, i == N - 1 ? '\n' : ',');
				data_[i] = std::stof(buffer);
			}

			if (hack_) normalizer_.normalize(data_);

			return false;
		}

	public:
		class iterator
		{
			csv_file& p_;
			bool end_;

			iterator(csv_file& p, bool end = false) : p_{ p }, end_{ end }
			{
				if (!end)
				{
					p_.file_.seekg(0);
					p_.forward();
				}
			}

		public:
			using value_type = matrix<T, N>;
			using reference = matrix<T, N>&;
			using pointer = matrix<T, N>*;
			using iterator_category = std::forward_iterator_tag;
			using difference_type = int;

			iterator  operator++() { end_ = p_.forward(); return *this; }
			iterator  operator++(int) { end_ = p_.forward(); return *this; }
			reference operator* () { return p_.data_; }
			pointer   operator->() { return &p_.data_; }
			bool      operator==(const iterator& rhs) { return &p_ == &rhs.p_ && end_ == rhs.end_; }
			bool      operator!=(const iterator& rhs) { return &p_ != &rhs.p_ || end_ != rhs.end_; }

			friend class csv_file;
		};

		csv_file(const std::string& path) : file_{ path }
		{
			for (const auto& row : *this) normalizer_.fit(row);
			hack_ = true;
		}

		auto begin() { return iterator(*this); }
		auto end() { return iterator(*this, true); }
		const auto begin() const { return iterator(*this); }
		const auto end()   const { return iterator(*this, true); }
	};
}