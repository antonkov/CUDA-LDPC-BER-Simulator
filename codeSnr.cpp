#include "simulator.h"
#include "filesystem.h"

#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <queue>

const float DEFAULT_TARGET_FER = 0.01;

struct settings_t
{
    float targetFER = DEFAULT_TARGET_FER;
} settings;

int main(int argc, char* argv[])
{
    simulation_params_t params;

    int opt;
    while ((opt = getopt(argc, argv, "f:")) != -1) {
        switch (opt) {
            case 'f':
                settings.targetFER = atof(optarg);
                break;
            default:
                printf("Usage: %s [-f targetFER] files*\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    std::queue<std::string> inputFilenames;
    for (int i = optind; i < argc; i++)
    {
        inputFilenames.push(argv[i]);
    }

    std::cout << "Results to get FER=" << settings.targetFER << std::endl;
    std::cout << "Filename Time(ms) SNR" << std::endl;

    while (!inputFilenames.empty())
    {
        std::string filename = inputFilenames.front();
        inputFilenames.pop();

        if (isDirectory(filename))
        {
            auto files = filesInDirectory(filename);
            for (auto file : files)
                inputFilenames.push(file);
            continue;
        }
        float result = 0;
        SimulationReport report;
        params.filename = filename;
        params.snr = 3;
        report = simulate(params);

        std::cout << filename << " ";
        std::cout << report.timeMs << " ";
        std::cout << result << " ";
        std::cout << std::endl;
    }
}
