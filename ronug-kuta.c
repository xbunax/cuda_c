#include <stdio.h>
#include <math.h>

double f(double x,double y) {
   double S;
   S=exp(x)+y;
    return S;
}

void runge_kutta(double x0, double x1,double y0, double h, int n) {
    double s1, s2, s3, s4, y;
    FILE *fp = fopen("output.txt", "w");
    for (int i = 0; i < n; i++) {
       s1 = f(x0, y0);
       s2 = f(x0 + h / 2,y0 + (h/2)*s1);
       s3 = f(x0 + h / 2,y0 + (h/2)*s2);
       s4 = f(x0 + h , y0 + h*s3);
        y = y0 + (s1 + 2 * s2 +2 *  s3 + s4)*h / 6;
        fprintf(fp, "%lf %lf\n", x0, y);
        y0 = y;
        x0 += h;
    }
    fclose(fp);
}

int main() {
    double x0 = 0, x1 = 5,y0 = 2, h = 0.1;
    int n = x1/h;
    runge_kutta(x0, x1, y0, h, n);
    return 0;
}