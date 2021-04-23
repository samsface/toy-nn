#include <catch2/catch.hpp>
#include "nn.hpp"
#include "nn_plot.hpp"
#include "test_util.hpp"

using namespace libai;

TEST_CASE("nn::activation::idenity::")
{
	nn::activation::idenity activation_function;

	SECTION("forward")
	{
		matrix<float_t, 1> m{ 1.0f };
		activation_function.forward(m);
		REQUIRE(m[0] == Approx(1.0f));
	}

	SECTION("back")
	{
		REQUIRE(activation_function.back(0.2f) == Approx(1.0f));
	}
}

TEST_CASE("nn::activation::sigmoid::")
{
	nn::activation::sigmoid activation_function;

	SECTION("forward")
	{
		matrix<float_t, 1> m{ 1.0f };
		activation_function.forward(m);
		REQUIRE(m[0] == Approx(0.7310585737f));
	}

	SECTION("back")
	{
		REQUIRE(activation_function.back(0.2f) == Approx(0.16f));
	}
}

TEST_CASE("nn::activation::relu::")
{
	nn::activation::relu activation_function;

	SECTION("forward")
	{
		matrix<float_t, 3> m{ -1.0f, 0.0f, 1.0f };
		activation_function.forward(m);
		REQUIRE(m[0] == Approx(0.0f));
		REQUIRE(m[1] == Approx(0.0f));
		REQUIRE(m[2] == Approx(1.0f));
	}

	SECTION("back")
	{
		REQUIRE(activation_function.back(-1.0f) == Approx(0.0f));
		REQUIRE(activation_function.back(+0.0f) == Approx(0.0f));
		REQUIRE(activation_function.back(+1.0f) == Approx(1.0f));
	}
}

TEST_CASE("nn::activation::softmax::")
{
	nn::activation::softmax activation_function;

	SECTION("forward")
	{
		matrix<float_t, 6> m{ 1.0f, 2.0f, 3.0f, 4.0f , 5.0f, 6.0f };
		activation_function.forward(m);

		float_t sum = std::accumulate(m.begin(), m.end(), 0.0f);
		REQUIRE(sum == Approx(1.0f));
	}

	SECTION("back")
	{
		REQUIRE(activation_function.back(0.2f) == Approx(0.16f));
	}
}

TEST_CASE("nn::loss::mean_squared_error::")
{
	nn::loss::mean_squared loss_function;

	SECTION("total")
	{
		matrix<float_t, 2> target{ 0.1f, 0.7f };
		matrix<float_t, 2> actual{ 0.5f, 0.1f };

		REQUIRE(loss_function.total(target, actual) == Approx(0.26f));
	}

	SECTION("prime")
	{
		matrix<float_t, 2> target{ 0.1f, 0.7f };
		matrix<float_t, 2> actual{ 0.5f, 0.1f };

		auto loss = loss_function.prime(target, actual);

		REQUIRE(loss[0] == Approx(-0.8f));
		REQUIRE(loss[1] == Approx(1.2f));
	}
}

TEST_CASE("nn::net::get", "[hide]")
{
	auto net = nn::net<
		nn::layer::dense<float_t, 1, 2>,
		nn::layer::dense<float_t, 2, 1>>{};

	SECTION("<0>")
	{
		nn::layer::dense<float_t, 1, 2>& l = net.get<0>();
	}

	SECTION("<1>")
	{
		nn::layer::dense<float_t, 2, 1>& l = net.get<1>();
	}

	SECTION("<-1>")
	{
		nn::layer::dense<float_t, 2, 1>& l = net.get<-1>();
	}
}

TEST_CASE("nn::net::sam2", "[hide]")
{
	auto net = nn::net<
		nn::layer::dense<float_t, 2, 2>,
		nn::layer::dense<float_t, 2, 2>>{};

	auto& in = net.get<0>();
	auto& out = net.get<1>();

	in.weights[0] = { 0.13436424411240122f, 0.84743373693723270f, 0.76377461897661400f };
	in.weights[1] = { 0.25506902573942170f, 0.49543508709194095f, 0.44949106478873810f };

	out.weights[0] = { 0.65159297272276300f, 0.78872335113551320f, 0.09385958677423490f };
	out.weights[1] = { 0.02834747652200631f, 0.83576510391986970f, 0.43276706790505337f };

	nn::optimizer::sgd<
		nn::loss::mean_squared,
		decltype(net)> optimizer;

	// some test data
	matrix<float_t, 1, 2> inputs;
	inputs[0] = { 2.781083600f, 2.550537003f };

	matrix<float_t, 1, 2> targets;
	targets[0] = { 1.0f, 0.0f };

	optimizer.fit(net, inputs, targets, 0.5f, 1);

	REQUIRE(in.outputs[0] == Approx(0.9643898158763548f));
	REQUIRE(in.outputs[1] == Approx(0.9185258960543243f));
	REQUIRE(out.outputs[0] == Approx(0.8094918973879515f));
	REQUIRE(out.outputs[1] == Approx(0.7734292563511262f));

	REQUIRE(in.deltas[0] == Approx(0.00105095399f));
	REQUIRE(in.deltas[1] == Approx(-0.01348571940f));
	REQUIRE(out.deltas[0] == Approx(0.05875834080f));
	REQUIRE(out.deltas[1] == Approx(-0.27106598000f));

	REQUIRE(in.weights[0][0] == Approx(0.135825649));
	REQUIRE(in.weights[0][1] == Approx(0.848774016));
	REQUIRE(in.weights[0][2] == Approx(0.764300108));

	REQUIRE(in.weights[1][0] == Approx(0.236316562));
	REQUIRE(in.weights[1][1] == Approx(0.478237182));
	REQUIRE(in.weights[1][2] == Approx(0.442748189));

	REQUIRE(out.weights[0][0] == Approx(0.679925919));
	REQUIRE(out.weights[0][1] == Approx(0.815708876));
	REQUIRE(out.weights[0][2] == Approx(0.123238757));

	REQUIRE(out.weights[1][0] == Approx(-0.102359161));
	REQUIRE(out.weights[1][1] == Approx(0.711274564));
	REQUIRE(out.weights[1][2] == Approx(0.297234058));
}

TEST_CASE("nn::net::keras_comparison", "[hide]")
{
	auto net = nn::net<
		nn::layer::dense<float_t, 1, 1>,
		nn::layer::dense<float_t, 1, 1>>{};

	net.get<0>().weights[0] = { 0.5f, 0.0f };
	net.get<1>().weights[0] = { 0.5f, 0.0f };

	auto optimizer = nn::optimizer::sgd<
		nn::loss::mean_squared,
		decltype(net)>{};

	matrix<float_t, 2, 1> inputs{ 0.5f, 0.5f };
	matrix<float_t, 2, 1> targets{ 1.0f, 1.0f };

	SECTION("defaults")
	{
		net.activate(inputs[0]);
		REQUIRE(net.output()[0] == Approx(0.569813f));

		optimizer.fit(net, inputs, targets, 1.0f, 1);
		net.activate(inputs[0]);
		REQUIRE(net.output()[0] == Approx(0.6881146));

		optimizer.fit(net, inputs, targets, 1.0f, 1);
		net.activate(inputs[0]);
		REQUIRE(net.output()[0] == Approx(0.7552961));
	}

	SECTION("mini_batch_size")
	{
		optimizer.fit(net, inputs, targets, 1.0f, 2);
		net.activate(inputs[0]);
		REQUIRE(net.output()[0] == Approx(0.63727844f));
	}

	SECTION("momentum")
	{
		optimizer.fit(net, inputs, targets, 1.0f, 1, 0.5f);
		net.activate(inputs[0]);
		REQUIRE(net.output()[0] == Approx(0.71790355));

		optimizer.fit(net, inputs, targets, 1.0f, 1, 0.5f);
		net.activate(inputs[0]);
		REQUIRE(net.output()[0] == Approx(0.8263175));
	}
}

TEST_CASE("pima-indians-diabetes", "[!hide]")
{
	test_util::csv_file<float_t, 9> csv_file{ "../test/data/pima-indians-diabetes.csv" };

	auto net = nn::net<
		nn::layer::dense<float_t, 8, 12, nn::activation::sigmoid>,
		nn::layer::dense<float_t, 12, 8, nn::activation::sigmoid>,
		nn::layer::dense<float_t, 8, 1, nn::activation::sigmoid>>{};

	nn::optimizer::sgd<
		nn::loss::mean_squared,
		decltype(net)> optimizer;

	nn::plot::plot plot{ net };

	matrix<float_t, 1, 8>  inputs;
	matrix<float_t, 1, 1>  targets;
	float_t loss = 0.0f;

	for (auto i = 0U; i < 1500; i++)
	{
		for (const auto& row : csv_file)
		{
			inputs[0] = row.reshape<8>();
			targets[0] = row.slice<8, 9>();

			optimizer.fit(net, inputs, targets, 0.01f, 1, 0.90f);
			loss += optimizer.total_loss(net, targets[0]);
		}

		plot.draw();
		std::cout << (loss / 768.0f) << std::endl;
		loss = 0.0f;
	}
}


TEST_CASE("xor", "[!hide]")
{
	auto net = nn::net<
		nn::layer::dense<float_t, 2, 2, nn::activation::relu>,
		nn::layer::dense<float_t, 2, 1>>{};

	nn::optimizer::sgd<
		nn::loss::mean_squared,
		decltype(net)> optimizer;

	nn::plot::plot plot{ net };

	for (int i = 0; i < 9200000; i++)
	{
		const auto mini_batch_size = 1U;

		matrix<float_t, mini_batch_size, 2> inputs;
		matrix<float_t, mini_batch_size, 1> targets;

		for (auto i = 0U; i < mini_batch_size; i++)
		{
			int a = rand() % 2;
			int b = rand() % 2;
			int r = a ^ b;
			inputs[i] = { static_cast<float_t>(a), static_cast<float_t>(b) };
			targets[i] = { static_cast<float_t>(r) };
		}

		optimizer.fit(net, inputs, targets, 0.001f, mini_batch_size, 0.99f);

		if (i % 1000 == 0)
		{
			plot.draw();
			const auto& o = net.output();

			if (o[0] > 0.9f && targets[mini_batch_size - 1][0] == 1.0f) std::cout << "correct";
			else if (o[0] < 0.1f && targets[mini_batch_size - 1][0] == 0.0f) std::cout << "correct";
			else std::cout << "wrong";

			std::cout << std::endl;
		}
	}
}

