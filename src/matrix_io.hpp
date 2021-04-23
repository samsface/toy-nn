#pragma once

#include <ostream>
#include "matrix.hpp"

template<typename T, size_t N>
inline std::ostream& operator<<(std::ostream& o, const gai::matrix<T, N>& m)
{
	o << "[ ";
	for (auto& i : m)
	{
		o << i << " ";
	}
	o << "]";

	return o;
}

template<size_t N>
inline std::ostream& operator<<(std::ostream& o, const gai::matrix<char, N>& m)
{
	o << "[ ";
	for (auto& i : m)
	{
		if(i == '\n')
		{
			o << "\\n";
		}
		else
		{
			o << i << " ";
		}
	}
	o << "]";

	return o;
}

template<typename T, size_t N, size_t M>
inline std::ostream& operator<<(std::ostream& o, const gai::matrix<T, N, M>& m)
{
	for (auto& i : m)
	{
		o << i << std::endl;
	}

    return o;
}
