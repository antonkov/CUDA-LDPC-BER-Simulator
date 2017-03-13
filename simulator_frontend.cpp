#include "simulator.h"
#include "filesystem.h"

#include <string>
#include <vector>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <queue>

const float DEFAULT_SNR = 3;

struct settings_t
{
    float snrFrom = DEFAULT_SNR;
    float snrTo = DEFAULT_SNR + 0.1;
    float snrStep = 1;
    bool runsTypeSet = false;
    float ferThreshold = 1e-4;
    // if FER lower than this value, don't calc further 
    // and print special value for rest of snrs
    // it can take too long
} settings;

int main(int argc, char* argv[])
{
    simulation_params_t params;

    int opt;
    while ((opt = getopt(argc, argv, "n:f:s:t:")) != -1) {
        switch (opt) {
            case 's':
                sscanf(optarg, "%f:%f:%f", &settings.snrFrom,
                        &settings.snrTo,
                        &settings.snrStep);
                break;
            case 'n': // Max number of codewords
                params.numberOfCodewords = atoi(optarg);
                break;
            case 'f': // Max number of Frame Errors
                params.numberOfFrameErrors = atoi(optarg);
                break;
            case 't':
                settings.ferThreshold = atof(optarg);
                break;
            default:
                printf("Usage: %s [-s snrFrom:snrTo:snrStep] [-n maxNumberOfCodewords] [-f maxNumberOfFrameErrors] [-t ferThreshold] files*\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    std::queue<std::string> inputFilenames;
    for (int i = optind; i < argc; i++)
    {
        inputFilenames.push(argv[i]);
    }

    std::cout << "Results" << std::endl;
    std::cout << "Filename SNR Time(ms) BER FER" << std::endl;

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

        Data data(filename);
        bool exceededThreshold = false;
        for (float snr = settings.snrFrom;
                snr < settings.snrTo;
                snr += settings.snrStep)
        {
            SimulationReport report;
            if (!exceededThreshold)
            {
                // Calling main simulation
                params.snr = snr;
                report = simulate(data, params);
            } else {
                report.BER = report.FER = -1;
                report.timeMs = 0;
            }

            std::cout << filename << " ";
            std::cout << snr << " ";
            std::cout << report.timeMs << " ";
            std::cout << report.BER << " ";
            std::cout << report.FER << " ";
            std::cout << std::endl;

            if (report.FER < settings.ferThreshold)
                exceededThreshold = true;
        }
    }
}
