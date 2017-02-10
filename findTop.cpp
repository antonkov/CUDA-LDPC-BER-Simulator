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
    float dbDiff = 0.05;
    int numberOfIters = 10;
    float searchL = 1;
    float searchR = 5;
} settings;

int main(int argc, char* argv[])
{
    simulation_params_t params;

    int opt;
    while ((opt = getopt(argc, argv, "l:r:f:i:d:")) != -1) {
        switch (opt) {
            case 'f':
                settings.targetFER = atof(optarg);
                break;
            case 'i':
                settings.numberOfIters = atoi(optarg);
                break;
            case 'd':
                settings.dbDiff = atof(optarg);
                break;
            case 'l':
                settings.searchL = atof(optarg);
                break;
            case 'r':
                settings.searchR = atof(optarg);
                break;
            default:
                printf("Usage: %s [-l snrL] [-r snrR] [-f targetFER] [-d dbDiff] [-i numberOfIters] files*\n",
                        argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    std::queue<std::string> inputFilenames;
    for (int i = optind; i < argc; i++)
    {
        inputFilenames.push(argv[i]);
    }

    std::vector<std::string> filenames;
    std::vector<float> searchL;
    std::vector<float> searchR;
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

        filenames.push_back(filename);
        searchL.push_back(settings.searchL);
        searchR.push_back(settings.searchR);
    }
    params.numberOfCodewords = 2 * params.numberOfFrameErrors / settings.targetFER;

    for (int iter = 0; iter < settings.numberOfIters; iter++)
    {
        std::cout << "Iter " << iter << " started" << std::endl;
        float bestR = 100;
        for (auto r : searchR)
            bestR = std::min(r, bestR);
        int good = 0;
        for (int i = 0; i < filenames.size(); i++)
        {
            if (searchL[i] - bestR > settings.dbDiff) {
                // This case is a lot worse than best one now
                std::cout << "iter " << iter << "  " << i
                    << " cont" << std::endl;
                continue;
            }
            good++;
            float snr = (searchL[i] + searchR[i]) / 2;

            params.filename = filenames[i];
            params.snr = snr;
            SimulationReport report = simulate(params);

            if (report.FER < settings.targetFER)
                searchR[i] = snr;
            else
                searchL[i] = snr;
            std::cout << "iter " << iter << " " << 
                filenames[i] << " " << snr << std::endl;
        }
        std::cout << "numberGood " << good << std::endl;
    }
}
