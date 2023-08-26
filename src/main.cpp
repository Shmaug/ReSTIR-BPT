#include "App/App.hpp"

int main(int argc, char** argv) {
	{
		ptvk::App app( std::span(argv, argc) | std::views::transform([](const char* a) { return std::string(a); }) | std::ranges::to<std::vector<std::string>>() );
		app.Run();
	}

	return EXIT_SUCCESS;
}