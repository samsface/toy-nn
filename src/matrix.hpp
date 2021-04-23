#pragma once

#include <array>
#include <limits>
#include <cstdint>
#include <cmath>
#include <queue>
#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>
#include "small_vector.hpp"

namespace gai
{
template<typename T, size_t N, size_t ...Tail>
class matrix
{
	template<size_t ...TailShadowed>
	static constexpr auto resolve_type_()
	{
		if constexpr (sizeof...(TailShadowed) == 0) return T{};
		else                                        return matrix<T, TailShadowed...>{};
	}

    template<typename ...TailShadowed>
    static constexpr size_t resolve_flat_size_(size_t s, TailShadowed ...t)
    {
        if constexpr(sizeof...(t) == 0) return s;
        else                            return s * resolve_flat_size_(t...);
    }

    using real_type_   = decltype(resolve_type_<Tail...>());
    using flat_size_   = std::integral_constant<size_t, resolve_flat_size_(N, Tail...)>;
    using flat_type_   = matrix<T, flat_size_::value>;

public:
	using value_type   = T;
	using dim_size     = std::integral_constant<size_t, 1 + sizeof...(Tail)>;
	using coord_type   = matrix<size_t, dim_size::value>;
	using fcoord_type_ = matrix<float_t, 1 + sizeof...(Tail)>;

	std::array<real_type_, N> data_;

	template<typename ...Args>
	constexpr matrix(Args const& ...args) : data_{ args... }
	{}

    constexpr size_t size()       const { return N; }
    constexpr size_t flat_size()  const { return flat_size_::value; }
	constexpr auto   dimensions() const { return matrix<size_t, sizeof...(Tail) + 1>{N, Tail...}; };

	constexpr auto& data()       { return flat().data_; }
	constexpr auto& data() const { return flat().data_; }

    constexpr auto& operator[](size_t i)       { return data_[i]; };
    constexpr auto& operator[](size_t i) const { return data_[i]; };

    constexpr auto& operator[](const matrix<size_t, dim_size::value>& coord)
    {
        if constexpr(sizeof...(Tail) == 0) return data_[coord[0]];
        else                               return data_[coord[0]][coord.template slice<1, dim_size::value>()];
    }

    constexpr auto& operator[](const matrix<size_t, dim_size::value>& coord) const
    {
        if constexpr(sizeof...(Tail) == 0) return data_[coord[0]];
        else                               return data_[coord[0]][coord.template slice<1, dim_size::value>()];
    }

    constexpr auto begin()       { return data_.begin(); }
    constexpr auto end()         { return data_.end();   }
    constexpr auto begin() const { return data_.begin(); }
    constexpr auto end()   const { return data_.end();   }

    constexpr bool operator==(matrix const& rhs) const
    {
		return data() == rhs.data();
    }

    constexpr bool operator!=(matrix const& rhs) const
    {
        return data() != rhs.data();
    }

    template<typename Op>
    constexpr matrix& mutable_op_(const matrix& rhs, Op op)
    {
        auto& lhsf = flatten();
        auto& rhsf = rhs.flatten();

        for(auto i = 0U; i < lhsf.size(); i++) lhsf[i] = op(lhsf[i], rhsf[i]);
        
        return *this;
    }

    template<typename Op>
    constexpr matrix imutable_op_(const matrix& rhs, Op op) const
    {
        auto  res  = flat_type_{};
        auto& lhsf = flatten();
        auto& rhsf = rhs.flatten();

        for(auto i = 0U; i < lhsf.size(); i++) res[i] = op(lhsf[i], rhsf[i]);

        return res;
    }

    template<typename Op>
    constexpr matrix& mutable_op_(const value_type& rhs, Op op)
    {
        auto& lhsf = flatten();
        for(auto i = 0U; i < lhsf.size(); i++) lhsf[i] = op(lhsf[i], rhs);
        
        return *this;
    }

    template<typename Op>
    constexpr matrix imutable_op_(const value_type& rhs, Op op) const
    {
        auto  res  = flat_type_{};
        auto& lhfs = flatten();

        for(auto i = 0UL; i < lhfs.size(); i++) res[i] = op(lhfs[i], rhs);

        return res;
    }

    // scalar value type arithmetic

    constexpr matrix& operator+=(const value_type& rhs) { return mutable_op_(rhs, std::plus{});  }
    constexpr matrix& operator-=(const value_type& rhs) { return mutable_op_(rhs, std::minus{}); }

    constexpr matrix& operator*=(const value_type& rhs) { return mutable_op_(rhs, std::multiplies{}); }
    constexpr matrix& operator/=(const value_type& rhs) { return mutable_op_(rhs, std::divides{});    }

    constexpr matrix operator*(const value_type& rhs) const { return imutable_op_(rhs, std::multiplies{}); }
    constexpr matrix operator/(const value_type& rhs) const { return imutable_op_(rhs, std::divides   {}); }

    constexpr matrix operator+(const value_type& rhs) const { return imutable_op_(rhs, std::plus {}); }
    constexpr matrix operator-(const value_type& rhs) const { return imutable_op_(rhs, std::minus{}); }

	// scalar self type arithmetic

	constexpr matrix& operator+=(const matrix& rhs) { return mutable_op_(rhs, std::plus{}); }
	constexpr matrix& operator-=(const matrix& rhs) { return mutable_op_(rhs, std::minus{}); }

	constexpr matrix operator+(const matrix& rhs) const { return imutable_op_(rhs, std::plus{}); }
	constexpr matrix operator-(const matrix& rhs) const { return imutable_op_(rhs, std::minus{}); }

	constexpr bool operator<(matrix const& rhs) const 
	{ 
		for(std::size_t i{}; i < data().size(); i++)
        {
            if(data()[i] < rhs.data()[i]) 
			{
				return true;
			}
        }

        return false; 
	}

    constexpr value_type sum() const
    {
		value_type res{};
        for(value_type const& i : flatten()) res += i;

        return res;
    }

	constexpr value_type abs_sum() const
    {
		value_type res{};
        for(value_type const& i : flatten()) res += std::abs(i);

        return res;
    }

    constexpr value_type magnitude() const
    {
		value_type res{};
        for(value_type const& i : flatten()) res += i*i;

        return std::sqrt(res);
    }

    constexpr value_type score() const
    {
        return magnitude() / std::sqrt(flat_size());
    }

    constexpr matrix& normalize()
    {
		value_type mag{magnitude()};
		if(mag > value_type{}) 
		{
        	for(value_type& i : flatten()) i /= mag;
		}

        return *this;
    }

	constexpr matrix normalized()
    {
		matrix res{*this};
		res.normalize();
		return res;
    }

    constexpr value_type dot_product(matrix const& rhs) const
    {
		value_type res{};

        auto const& a = flatten();
        auto const& b = rhs.flatten();

		for(size_t i{}; i < size(); i++)
        {
            res += a[i] * b[i];
        }

        return res; 
    }

	value_type distance_to(matrix const& rhs) const
	{
		value_type res{};
	
		for(std::size_t i{}; i < data().size(); i++)
		{
			res += std::pow(rhs.data()[i] - data()[i], 2);
		}

		return sqrt(res);
	}
    /*
    template<size_t FN>
    auto convolve(const matrix<T, FN, FN>& filter) const
    {
        const size_t stride = 2;

        matrix<T, XN / stride, YN / stride> res;

        for(size_t s_y = 0; s_y < YN; s_y++)
        for(size_t s_x = 0; s_x < XN; s_x++)
        {
            auto sum = 0;
            for(size_t f_y = 0; f_y < FN && s_y + f_y < YN; f_y++)
            for(size_t f_x = 0; f_x < FN && s_x + f_x < XN; f_x++)
            {
                sum += filter[f_y][f_x] * data_[s_y + f_y][s_x + f_x];                
            }

            sum /= (FN*FN);
            
            // pool by max
            res[s_y / stride][s_x / stride] = std::max(res[s_y / stride][s_x / stride], sum);
        }

        return res;
    }
    */

    constexpr flat_type_      & flatten()       { return reinterpret_cast<flat_type_      &>(*data_.data()); }
    constexpr flat_type_ const& flatten() const { return reinterpret_cast<flat_type_ const&>(*data_.data()); }

    constexpr flat_type_      & flat()       { return reinterpret_cast<flat_type_      &>(*data_.data()); }
    constexpr flat_type_ const& flat() const { return reinterpret_cast<flat_type_ const&>(*data_.data()); }

    template<size_t ...ReshapedTail>
    constexpr auto& reshape()
    {
        static_assert(resolve_flat_size_(ReshapedTail...) <= flat_size_::value, "reshape is out of bounds");
        
        return reinterpret_cast<matrix<value_type, ReshapedTail...>&>(*data_.data());
    }

    template<size_t ...ReshapedTail>
    constexpr auto const& reshape() const
    {
        static_assert(resolve_size_(ReshapedTail...) <= flat_size_::value, "reshape is out of bounds");
        
        return reinterpret_cast<const matrix<value_type, ReshapedTail...>&>(*data_.data());
    }

    template<size_t X, size_t Y>
    constexpr auto& slice()
    {
        static_assert(sizeof...(Tail) == 0, "slice only allowed on 1D matrix.");
        static_assert(Y > X,                "Y must be greater than X");
        static_assert(Y <= N,               "Y must be less than or equal to N");

        return reinterpret_cast<matrix<value_type, (Y-X)>&>(*(flatten().begin() + X));
    }

    template<size_t X, size_t Y>
    constexpr auto const& slice() const
    {
        static_assert(sizeof...(Tail) == 0, "slice only allowed on 1D matrix.");
        static_assert(Y > X,                "Y must be greater than X");
        static_assert(Y <= N,               "Y must be less than or equal to N");

        return reinterpret_cast<const matrix<value_type, (Y-X)>&>(*(flatten().begin() + X));
    }

    template<typename cast_type>
    constexpr matrix<cast_type, N, Tail...> cast() const
    {
        matrix<cast_type, flat_size_::value> res;
        const auto& lhsf = flatten();

        for(auto i = 0UL; i < lhsf.size(); i++)
        {
            res[i] = static_cast<cast_type>(lhsf[i]);
        }

        return res;
    }

	auto max_element() const
	{
		return *std::max_element(flatten().begin(), flatten().end());
	}

	auto max_abs_element() const
	{
		return *std::max_element(flatten().begin(), flatten().end(), [](value_type const& rhs, value_type const& lhs){ return std::abs(rhs) < std::abs(lhs); });
	}

	// weird stuff

	constexpr void fill(value_type const& t)
	{
		flatten().data_.fill(t);
	}

	constexpr coord_type fit(coord_type const& coord) const
	{
		coord_type res;

		auto const dims = dimensions();
		for(std::size_t i{}; i < coord.size(); i++)
		{
			std::size_t f{coord[i]};
			f = std::min(f, dims[i] - std::size_t{1});
			f = std::max(f, std::size_t{});
			res[i] = f;
		}

		return res;
	}

	constexpr coord_type fit(fcoord_type_ const& fcoord) const
	{
		coord_type res;

		auto const dims = dimensions().template cast<float_t>();
		for(size_t i{}; i < fcoord.size(); i++)
		{
			float_t f{fcoord[i]};
			f = std::min(f, dims[i] - 1);
			f = std::max(f, 0.0f);
			res[i] = static_cast<size_t>(f);
		}

		return res;
	}

	constexpr bool fits(coord_type const& coord) const
	{
		auto const dims = dimensions();
		for(size_t i{}; i < coord.size(); i++)
		{
			if(coord[i] >= dims[i])
			{
				return false;
			}
		}

		return true;
	}

	auto rotate(matrix<float_t, 2> val, float_t angle) const
	{
		return val;
	}

	auto flip()
	{

	}
};

template<typename T, typename TT>
auto rotate(matrix<T, 2> val, float_t angle, matrix<TT, 2> origin)
{
	angle /= 57.29578f;

	matrix<float_t, 2> o{origin.template cast<float_t>()};
	matrix<float_t, 2> f{val   .template cast<float_t>()};
	f -= o;

	matrix<float_t, 2> res{
		(f[1] * std::sin(angle)) + (f[0] * std::cos(angle)), 
		(f[1] * std::cos(angle)) - (f[0] * std::sin(angle))};

	return res + o;
}

template<typename T, size_t ...N>
T manhattan_distance(matrix<T, N...> const& a, matrix<T, N...> const& b)
{
    auto& af = a.flatten();
    auto& bf = b.flatten();

    T res{};

    for(size_t i{}; i < af.size(); i++) res += std::abs(static_cast<signed>(af[i]) - static_cast<signed>(bf[i]));

    return res;
}

template<typename T> using vec2_t = matrix<T, 2>;	
template<typename T> using vec3_t = matrix<T, 3>;
using uvec2_t = vec2_t<size_t>;
using uvec3_t = vec3_t<size_t>;
using fvec2_t = vec2_t<float_t>;
using ivec2_t = vec2_t<int>;

template<typename T> using umat_t = matrix<T, 4, 4>;

namespace cardinal 
{
#pragma warning( push )
#pragma warning( disable: 4146 )

	uvec2_t const center     = { 0UL,  0UL};
	uvec2_t const north      = { 1UL,  0UL};
	uvec2_t const north_east = { 1UL,  1UL};
	uvec2_t const east       = { 0UL,  1UL};
	uvec2_t const south_east = { static_cast<size_t>(-1), static_cast<size_t>( 1) };
	uvec2_t const south =      { static_cast<size_t>(-1), static_cast<size_t>( 0) };
	uvec2_t const south_west = { static_cast<size_t>(-1), static_cast<size_t>(-1) };
	uvec2_t const west       = { static_cast<size_t>( 0), static_cast<size_t>(-1) };
	uvec2_t const north_west = { static_cast<size_t>( 1), static_cast<size_t>(-1) };

#pragma warning( pop )

	uvec2_t const compass[] =
	{
		north, east, west, south
	};

	uvec2_t const compass_ex[] =
	{
		north_west, north, north_east, west, east, south_west, south, south_east
	};

	uvec2_t const compass_ex_with_center[] =
	{
		north_west, north, north_east, west, center, east, south_west, south, south_east
	};

	uvec2_t const clockwise_compass[] =
	{
		north, north_east, east, south_east, south, south_west, west, north_west, north, north_east
	};
}

template<typename T, size_t ...dimensions>
class line_move
{
	using matrix_type  =          matrix<T, dimensions...>;
	using coord_type   = typename matrix_type::coord_type;
	using fcoord_type_ = typename matrix_type::fcoord_type_;

	matrix_type& mat_;
	fcoord_type_ begin_;
	fcoord_type_ end_;
	fcoord_type_ delta_;
	size_t       step_;

public:
	class iterator
	{
		line_move&   line_generator_;
		size_t       step_;
		fcoord_type_ position_;

		iterator(line_move& line_generator, size_t step, fcoord_type_ const& position) :
			line_generator_{line_generator},
			step_          {step},
			position_      {position}
		{}

		void forward()
		{
			auto const next = position_ + line_generator_.delta_; // fix this right in ctor you idiot
			if(line_generator_.mat_.fits(next.template cast<std::size_t>()))
			{
				position_ = next;
			}

			step_++;
		}

	public:
		using value_type = T;
		using reference  = T&;
		using pointer    = T*;

				  iterator  operator++()                          { forward(); return *this; }
				  iterator  operator++(int)                       { forward(); return *this; }
				  reference operator* ()                          { return  line_generator_.mat_[position_.template cast<size_t>()]; }
				  pointer   operator->()                          { return &line_generator_.mat_[position_.template cast<size_t>()]; }
		constexpr bool      operator==(iterator const& rhs) const { return step_ == rhs.step_; }
		constexpr bool      operator!=(iterator const& rhs) const { return step_ != rhs.step_; }

		friend class line_move;
	};

	line_move(matrix_type& mat, coord_type const& begin, coord_type const& end) :
		mat_  {mat},
		begin_{begin.template cast<float_t>() + 0.5f},
		end_  {end  .template cast<float_t>()},
		delta_{end_ - begin_},
		step_ {static_cast<size_t>(1 + std::abs(delta_.max_abs_element()))}
	{
		delta_ /= static_cast<float_t>(step_);
	}

	line_move(matrix_type& mat, fcoord_type_ const& begin, fcoord_type_ const& end) :
		mat_  {mat},
		begin_{begin + 0.5f},
		end_  {end},
		delta_{end_ - begin_},
		step_ {static_cast<size_t>(1 + std::abs(delta_.max_abs_element()))}
	{
		delta_ /= static_cast<float_t>(step_);
	}

	line_move(matrix_type& mat, coord_type const& begin, iterator const& end) :
		line_move{mat, begin, end.position_.template cast<size_t>()}
	{}

			  auto begin()       { return iterator{*this, 0,     begin_}; }
			  auto end()         { return iterator{*this, step_, end_  }; }
	constexpr auto begin() const { return iterator{*this, 0,     begin_}; }
	constexpr auto end()   const { return iterator{*this, step_, end_  }; }
};


template<typename T, typename Op, size_t ...dimensions>
void cast_arc(matrix<T, dimensions...>& mat, typename matrix<T, dimensions...>::coord_type const& pos, size_t distance, size_t arc, size_t rotation, Op op)
{
	int step = 4;
	int r = arc / step;

	for(int i = {-r/2}; i < (r / 2); i++)
	{
		auto end = rotate(pos + uvec2_t{0ul, distance}, (step * i) + static_cast<int>(rotation), pos);

		for (auto& t: line_move{mat, pos.template cast<float_t>(), end})
		{
			if(!op(t))
			{
				break;
			}
		}
	}
}

template<typename T, typename score_op, size_t max_path_size = 256, size_t ...dimensions>
auto find_path(matrix<T, dimensions...> const& mat, typename matrix<T, dimensions...>::coord_type const& begin, typename matrix<T, dimensions...>::coord_type const& end, score_op score)
{
	using self_type   = matrix<T, dimensions...>;
	using coord_type  = typename self_type::coord_type;
	using weight_type = decltype(score(coord_type{}));
	
	struct nav_node
	{
		enum class state
		{
			open,
			queued,
			closed
		};

		nav_node*   parent{};
		coord_type  position;
		weight_type score{};
		state       state{};

		nav_node() = default;

		nav_node(nav_node* parent, coord_type position, weight_type score) :
			parent{ parent },
			position{ position },
			score{ score }
		{}

		bool empty() const
		{
			return parent == nullptr;
		}
	};

	struct nav_node_ptr_compare
	{
		bool operator()(nav_node const* a, nav_node const* b) const
		{
			return a->score > b->score;
		}
	};

	small_vector<coord_type, max_path_size> res;

	if (begin == end) return res;

	matrix<nav_node, dimensions...> nav_nodes{};
	std::priority_queue<nav_node*, std::vector<nav_node*>, nav_node_ptr_compare> open_nodes;

	size_t limit{ 1000 };

	nav_nodes[end] = nav_node{ nullptr, end, weight_type{} };
	open_nodes.push(&nav_nodes[end]);

	while (open_nodes.size() && limit--)
	{
		nav_node* current_node = open_nodes.top();
		open_nodes.pop();

		current_node->state = nav_node::state::closed;

		for (coord_type const d : cardinal::compass)
		{
			coord_type const exploring_position = current_node->position + d;

			if (!mat.fits(exploring_position)) continue;

			nav_node& exploring_node = nav_nodes[exploring_position];

			if (exploring_node.state == nav_node::state::closed) continue;

			weight_type const move_score = score(exploring_position) + current_node->score;

			if (exploring_node.empty())
			{
				if (exploring_position == begin)
				{
					nav_node const* p = current_node;

					res.push_back(exploring_position);
					do { res.push_back(p->position); } while ((p = p->parent));

					return res;
				}

				exploring_node = nav_node{ current_node, exploring_position, move_score };
			}

			if (exploring_node.score > move_score)
			{
				// this doesn't reheap
				// need to fix somehow
				exploring_node.parent = current_node;
				exploring_node.score  = move_score;
			}

			if (exploring_node.state == nav_node::state::open)
			{
				exploring_node.state = nav_node::state::queued;
				open_nodes.push(&exploring_node);
			}
		}
	}

	return res;
}

template<typename T, typename score_op, size_t ...dimensions>
auto make_field(matrix<T, dimensions...> const& mat, matrix<size_t, sizeof...(dimensions)> const& begin, score_op score)
{
	using self_type   = matrix<T, dimensions...>;
	using coord_type  = typename self_type::coord_type;
	using weight_type = decltype(score(coord_type{}));

	struct node
	{
		enum class state
		{
			open,
			queued,
			closed
		};

		state      state   {};
		coord_type position{};
	};

	small_queue<node*, 256>                                                open_nodes;
	matrix<node, dimensions...>                                            nodes;
	matrix<matrix<weight_type, self_type::dim_size::value>, dimensions...> weights;

	node* begin_node     = &nodes[begin];
	begin_node->state    = node::state::queued;
	begin_node->position = begin;

	open_nodes.push(begin_node);

	while (!open_nodes.empty())
	{
		node* current = open_nodes.front();
		open_nodes.pop();

		for (coord_type const& d : cardinal::compass)
		{
			coord_type anp = current->position + d;
			if (!mat.fits(anp)) continue;

			node* adj_node = &nodes[anp];

			if (adj_node->state != node::state::closed)
			{
				weight_type s = score(anp);

				if (s <= 0.0f)
				{
					adj_node->state = node::state::closed;
				}
				else
				{
					// + 1 to avoid the overflow
					weights[anp]      += ((d + 1).template cast<weight_type>() - 1.0f) * s;
					weights[anp].normalize();

					if (adj_node->state == node::state::open)
					{
						adj_node->position = anp;
						adj_node->state = node::state::queued;
						open_nodes.push(adj_node);
					}
				}
			}
		}

		current->state = node::state::closed;
	}

	return weights;
}

template<typename T, typename score_op, size_t ...dimensions>
void for_each_neighbour(matrix<T, dimensions...>& mat, score_op score)
{
	for(std::size_t y{}; y < mat.size(); y++)
	{
		for(std::size_t x{}; x < mat[y].size(); x++)
		{
			for(auto& direction : cardinal::compass_ex)
			{
				auto c = direction + uvec2_t{y, x};

				if(mat.fits(c)) 
				{
					score(mat[y][x], mat[c]);
				}
			}
		}
	}
};

template<typename T, typename score_op, size_t ...dimensions>
auto transform_neighbour(matrix<T, dimensions...> const& mat, score_op score)
{
	matrix<T, dimensions...> res = mat;

	for(std::size_t y{}; y < mat.size(); y++)
	{
		for(std::size_t x{}; x < mat[y].size(); x++)
		{
			for(auto& direction : cardinal::compass_ex)
			{
				auto c = direction + uvec2_t{y, x};

				if(mat.fits(c)) 
				{
					res[y][x] = score(mat[y][x], mat[c]);
				}
			}
		}
	}

	return res;
};

template<typename T, typename score_op, size_t ...dimensions>
auto neighbour(matrix<T, dimensions...> const& mat, score_op score)
{
	matrix<T, dimensions...> res;

	for(std::size_t y{}; y < mat.size(); y++)
	{
		for(std::size_t x{}; x < mat[y].size(); x++)
		{
			std::size_t neighbour_count{};

			for(auto& direction : cardinal::compass_ex)
			{
				auto c = direction + uvec2_t{y, x};

				if(mat.fits(c) && mat[c]) 
				{
					neighbour_count++;
				}
			}

			res[uvec2_t{y, x}] = score(neighbour_count);
		}
	}

	return res;
};

template<typename T, size_t N>
auto rotate_elements(matrix<T, N, N> const& mat)
{
	matrix<T, N, N> res;

	for(std::size_t y{}; y < res   .size(); y++)
	for(std::size_t x{}; x < res[y].size(); x++)
	{
		res[y][x] = mat[res[y].size() - x - 1][y];
	}

	/*
	for(std::size_t i{}; i < mat.data().size(); i++)
	{
		std::size_t const y{i / N};
		std::size_t const x{i % N};
		res.data()[i] = mat.data()[(N - 1 - x) * N + (y % N)];
	}
	*/

	return res;
}
}
