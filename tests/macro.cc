#define MACRO_TEST(X, Y) \
{                       \
    return (__LINE__);     \
}

int main() {
    MACRO_TEST(1, 1);
}