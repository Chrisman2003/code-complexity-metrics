int square(int n) {
    return n * n;
}

int process(int x) {
    int y = 0;
    if (x > 0)
        y = square(x);
    else
        y = -1;
    return y;
}