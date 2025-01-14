#include <stdio.h>

int main() {
    int n, i, j;
    float x[100], y[100], a[100], p, v, xv;
    printf(" sajan bista newton's divided difference\n");

    printf("Enter the number of data points: ");
    scanf("%d", &n);
    
    printf("Enter the value of x for interpolation: ");
    scanf("%f", &xv);
    
    
    for (i = 0; i < n; i++) {
        printf("Enter the data points x, and f(x) at i=%d",i);
        scanf("%f %f", &x[i], &y[i]);
    }

    

    // Initialize divided difference table
    for (i = 0; i < n; i++) {
        a[i] = y[i];
    }

    // Calculate divided differences
    for (i = 1; i < n; i++) {
        for (j = n - 1; j >= i; j--) {
            a[j] = (a[j] - a[j - 1]) / (x[j] - x[j - i]);
        }
    }

    // Calculate interpolated value
    p = 1;
    v = a[n - 1];
    for (i = n - 2; i >= 0; i--) {
        p *= (xv - x[i]);
        v += a[i] * p;
    }

    printf("Interpolated value  = %f\n", v);

    return 0;
}
