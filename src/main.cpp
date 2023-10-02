#include "App/App.hpp"

int main(int argc, char** argv) {
	std::vector<std::string> args(argc);
	for (uint i = 0; i < argc; i++)
		args[i] = argv[i];

	ptvk::App app( args );
	
	app.Run();

	return EXIT_SUCCESS;
}