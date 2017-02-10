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
    int numberOfIters = 10;
} settings;

int main(int argc, char* argv[])
{
    simulation_params_t params;

    int opt;
    while ((opt = getopt(argc, argv, "f:i:")) != -1) {
        switch (opt) {
            case 'f':
                settings.targetFER = atof(optarg);
                break;
            case 'i':
                settings.numberOfIters = atoi(optarg);
                break;
            default:
                printf("Usage: %s [-f targetFER] [-i numberOfIters] files*\n",
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

        params.filename = filename;
        params.numberOfCodewords = 2 * params.numberOfFrameErrors / settings.targetFER;
        // Don't need to test more codewords because probability 
        // that we get less that 100 errors when we expect to get 200
        // if numberOfFrameErrors = 100 then 3sigma = 0.3 and we have 0.5
        // if numberOfFrameErrors = 50 then 3sigma = 0.42 and we have 0.5
        // in 3 sigma 0.999
        // in 4 sigma 0.99997
        // in 5 sigma 0.999999997 and one run takes roughly 0.1s
        // expected time to get error with 5 sigma 300 million seconds
        // or 570 years or running ;) 

        float timeSum = 0;
        float snrFrom = 1, snrTo = 5;
        SimulationReport report;
        for (int iter = 0; iter < settings.numberOfIters; iter++)
        {
            float snrMid = (snrFrom + snrTo) / 2;
            params.snr = snrMid;
            report = simulate(params);

            if (report.FER < settings.targetFER)
                snrTo = snrMid;
            else
                snrFrom = snrMid;

            timeSum += report.timeMs;
        }
        float result = snrTo;

        std::cout << filename << " ";
        std::cout << timeSum << " ";
        std::cout << result << " ";
        std::cout << std::endl;
    }
}
