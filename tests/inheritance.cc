#include <iostream>

using namespace std;

class Base {
 public:
    virtual void func() {
        cout << "Base::func" << endl;
    }

    virtual ~Base() {
        cout << "~Base()" << endl;
    }

    int a = 4;
};

class Base2 {
public:
    virtual void func() {
        cout << "Base2::func" << endl;
    }

    virtual void func3() {

    }
};

class Derived : public Base, public Base2 {
 public:
    void func2() {
        cout << "Derived::func" << endl;
    }
    ~Derived() {
        cout << "~Derived" << endl;
    }

    int a = 3;
};

int main() {
    Base2 *d = new Derived;
    d->func3();
    cout << d << endl;
}