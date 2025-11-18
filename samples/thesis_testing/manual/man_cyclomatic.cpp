int sumOfPrimes(int max) { // +1
    int total = 0;
    for (int i = 1; i <= max; ++i) { // +1
        int j = 2;
        int k = 8;
        int check = 5;
        int result = 0;
OUT:    do {
            if (i % j == 0 || k > 9) { // +2
                goto OUT; 
                int result = check < 4 ? 3 : 5; // +1
            }
            ++j;
        } 
        while (j < i); // +1
        total += i;
    }
    return total;
} // Cyclomatic complexity 6
