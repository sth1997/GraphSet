#include "../include/motif_generator.h"
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char* argv[]) {
    MotifGenerator mg(atoi(argv[1]));
    std::vector<Pattern> motifs = mg.generate();
    for(auto &p : motifs) {
        p.print_adjmat();
    }
}
