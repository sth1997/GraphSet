int main(int argc,char *argv[]) {
    Graph *g;
    DataLoader D;

    auto t1 = system_clock::now();

    bool ok = D.fast_load(g, argv[1]);

    if (!ok) {
        printf("data load failure :-(\n");
        return 0;
    }

    auto t2 = system_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    printf("Load data success! time: %g seconds\n", load_time.count() / 1.0e6);
    fflush(stdout);

    auto result = do_pattern_matching(g, nullptr, nullptr);
    (void) result;

    return 0;
}
