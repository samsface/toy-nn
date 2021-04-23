#pragma once

#include <numeric>
#include "matrix.hpp"
#include "tuple_util.hpp"

namespace libai::nn::activation
{
	struct idenity
	{
		template<typename T, size_t N>
		void forward(matrix<T, N>& m)
		{}

		template<typename T>
		auto back(T t)
		{
			return 1;
		}
	};

	struct sigmoid
	{
		template<typename T, size_t N>
		void forward(matrix<T, N>& m)
		{
			for (auto& t : m) t = 1 / (1 + exp(-t));
		}

		template<typename T>
		auto back(T t)
		{
			return t * (1 - t);
		}
	};

	struct relu
	{
		template<typename T, size_t N>
		void forward(matrix<T, N>& m)
		{
			for (auto& t : m) t = std::max(t, T{});
		}

		template<typename T>
		auto back(T t)
		{
			return t <= 0 ? 0 : 1;
		}
	};

	struct softmax
	{
		template<typename T, size_t N>
		void forward(matrix<T, N>& m)
		{
			T sum{};
			for (auto& t : m) { t = exp(t); sum += t; }
			for (auto& t : m) t /= sum;
		}

		template<typename T>
		auto back(T t)
		{
			return t * (1 - t);
		}
	};
} // activations

namespace libai::nn::loss
{
	struct mean_squared
	{
		template<typename T, size_t N>
		auto total(const matrix<T, N>& target, const matrix<T, N>& actual)
		{
			T loss_sum{};

			for (auto i = 0U; i < N; i++)
			{
				loss_sum += pow(target[i] - actual[i], 2);
			}

			return loss_sum / N;
		}

		template<typename T, size_t N>
		auto prime(const matrix<T, N>& target, const matrix<T, N>& actual)
		{
			matrix<T, N> losses{};

			for (auto i = 0U; i < N; i++)
			{
				losses[i] = 2 * (target[i] - actual[i]);
			}

			return losses;
		}
	};

	struct cross_entropy
	{
		template<typename T, size_t N>
		auto total_error(const matrix<T, N>& target, const matrix<T, N>& actual)
		{
			T loss_sum{};

			for (auto i = 0U; i < actual.size(); i++)
			{
				loss_sum += pow(target[i] - actual[i], 2);
			}

			return loss_sum / N;
		}

		template<typename T, size_t N>
		auto prime(const matrix<T, N>&, const matrix<T, N>& actual)
		{
			matrix<T, N> losses{};

			for (auto i = 0U; i < N; i++)
			{
				losses[i] = -1 / actual[i];
			}

			return losses;
		}
	};
} // loss

namespace libai::nn::layer
{
	template<typename T, size_t input_count, size_t output_count, typename activation_op = activation::sigmoid>
	struct dense
	{
		using type = T;
		using input_t = matrix<type, input_count>;
		using output_t = matrix<type, output_count>;
		using weight_t = matrix<type, output_count, input_count + 1>;

		matrix<type, output_count, input_count + 1> weights{};
		matrix<type, output_count>                  outputs{};
		matrix<type, output_count>                  deltas{};
		matrix<type, output_count, input_count + 1> gradients{};

		dense()
		{
			for (auto& weight : weights.flatten())
			{
				weight = ((float_t)rand() / RAND_MAX);
			}
		}

		void activate(const matrix<type, input_count>& inputs)
		{
			outputs = {};

			for (auto w = 0U; w < output_count; w++)
			{
				for (auto i = 0U; i < input_count; i++)
				{
					outputs[w] += inputs[i] * weights[w][i];
				}

				outputs[w] += weights[w][input_count];
			}

			activation_op{}.forward(outputs);
		}

		void backy(const matrix<type, output_count>& losses, const matrix<type, input_count>& inputs)
		{
			for (auto w = 0U; w < output_count; w++)
			{
				deltas[w] = losses[w] * activation_op{}.back(outputs[w]);
			}

			for (auto w = 0U; w < weights.size(); w++)
			{
				for (auto i = 0U; i < inputs.size(); i++)
				{
					gradients[w][i] += deltas[w] * inputs[i];
				}

				gradients[w][inputs.size()] += deltas[w];
			}
		}

		template<size_t weights_in_count>
		void backtivate(const matrix<type, weights_in_count, output_count + 1>& weights_in,
			const matrix<type, weights_in_count>& deltas_in,
			const matrix<type, input_count>& inputs)
		{
			matrix<type, output_count> losses{};

			for (auto w = 0U; w < output_count; w++)
			{
				for (auto pw = 0U; pw < weights_in_count; pw++)
				{
					losses[w] += weights_in[pw][w] * deltas_in[pw];
				}
			}

			backy(losses, inputs);
		}
	};

	template<typename T, size_t input_count, size_t output_count, typename activation_op = activation::relu>
	struct convolve
	{
		using type = T;
		using input_t = matrix<type, input_count>;
		using output_t = matrix<type, output_count>;
		using weight_t = matrix<type, output_count, input_count>;

		matrix<type, output_count, input_count + 1> weights{};
		matrix<type, output_count>                  outputs{};
		matrix<type, output_count>                  deltas{};
		matrix<type, output_count, input_count + 1> gradients{};

		void activate(const matrix<type, input_count>& inputs)
		{
			outputs = {};

			for (auto w = 0U; w < output_count; w++)
			{
				for (auto i = 0U; i < input_count; i++)
				{
					// outputs[w] += inputs[s].convolve(weights[w][i]);
				}
			}
		}
	};
} // layer

namespace libai::nn
{
	template<typename ...Layers>
	struct net
	{
		std::tuple<Layers...> layers_;

		template<typename ...Args>
		net(Args&& ...args) : layers_{ std::forward<Args>(args)... }
		{}

		template<int I>       auto& get() { return sget<I>(layers_); }
		template<int I> const auto& get() const { return sget<I>(layers_); }

		auto size() const { return sizeof...(Layers); }

		const auto& output() const { return get<-1>().outputs; }

		template<typename T>
		void activate(const T& inputs)
		{
			get<0>().activate(inputs);
			for_each_2(layers_, [](const auto& left, auto& right) { right.activate(left.outputs); });
		}

		template<typename L, typename I>
		void backtivate(const L& losses, const I& inputs)
		{
			get<-1>().backy(losses, get<-2>().outputs);

			for_each_3_reverse(layers_,
				[](const auto& left, auto& middle, const auto& right)
			{
				middle.backtivate(right.weights, right.deltas, left.outputs);
			});

			get<0>().backtivate(get<1>().weights, get<1>().deltas, inputs);
		}
	};
} // nn

namespace libai::nn::optimizer
{
	template<typename Loss, typename Net>
	struct sgd
	{};

	template<typename Loss, typename ...Layers>
	struct sgd<Loss, net<Layers...>>
	{
		template <typename Layer>
		struct meta_layer_
		{
			typename Layer::weight_t moments{};
		};

		Loss loss_;
		std::tuple<meta_layer_<Layers>...> meta_layers_;

		template<typename Targets>
		auto total_loss(const net<Layers...>& net, const Targets& targets)
		{
			return loss_.total(targets, net.output());
		}

		void fit_(net<Layers...>& net, float_t learning_rate, float_t momentum, size_t mini_batch_size)
		{
			for_each([&](auto& layer, auto& meta_layer)
			{
				auto& weights = layer.weights.flatten();
				auto& gradients = layer.gradients.flatten();
				auto& moments = meta_layer.moments.flatten();

				for (auto i = 0U; i < weights.size(); i++)
				{
					moments[i] = (momentum * moments[i]) + (learning_rate * (gradients[i] / mini_batch_size));
					weights[i] += moments[i];
				}

				gradients = {};
			},
				net.layers_, meta_layers_);
		}

		template<typename I, typename T>
		void fit(net<Layers...>& net, const I& inputs, const T& targets, float_t learning_rate = 0.01f, size_t mini_batch_size = 1, float_t momentum = 0.0f)
		{
			for (auto i = 0; i < inputs.size(); i++)
			{
				net.activate(inputs[i]);
				net.backtivate(loss_.prime(targets[i], net.template get<-1>().outputs), inputs[i]);

				if (i % mini_batch_size == mini_batch_size - 1)
				{
					fit_(net, learning_rate, momentum, mini_batch_size);
				}
			}
		}
	};
} // optimizer
