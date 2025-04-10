## 1. LL(1) Parser for Arithmetic Expressions

### **Description**
This program implements an LL(1) parser for expressions of the form `E -> T + E | T` and `T -> id`.

### **Code (ll1_parser.c)**
```c
#include <stdio.h>
#include <string.h>

char input[100];
int pos = 0;

void E();
void T();

void match(char expected) {
    if (input[pos] == expected)
        pos++;
    else {
        printf("Syntax Error!\n");
        exit(0);
    }
}

void T() {
    if (input[pos] == 'a') {  // Assume 'a' is an identifier
        match('a');
    } else {
        printf("Syntax Error in T!\n");
        exit(0);
    }
}

void E() {
    T();
    if (input[pos] == '+') {
        match('+');
        E();
    }
}

int main() {
    printf("Enter an expression (e.g., a+a): ");
    scanf("%s", input);
    E();
    if (input[pos] == '\0')
        printf("Parsing successful!\n");
    else
        printf("Syntax Error at end!\n");
    return 0;
}
```

### **Compilation & Execution**
```bash
gcc ll1_parser.c -o parser
./parser
```

### **Sample Input**
```
a+a
```

### **Output**
```
Enter an expression (e.g., a+a): a+a
Parsing successful!
```

---

## 2. Loop Unrolling Program

### **Description**
This program demonstrates loop unrolling, a compiler optimization technique where loop iterations are expanded to reduce branch overhead.

### **Code (loop_unrolling.c)**
```c
#include <stdio.h>

void loop_unrolling() {
    int i;
    printf("Unrolled loop output:\n");
    for (i = 0; i < 10; i += 2) {  // Unrolling factor of 2
        printf("%d ", i);
        if (i + 1 < 10) printf("%d ", i + 1);
    }
    printf("\n");
}

int main() {
    loop_unrolling();
    return 0;
}
```

### **Compilation & Execution**
```bash
gcc loop_unrolling.c -o loop
./loop
```

### **Output**
```
Unrolled loop output:
0 1 2 3 4 5 6 7 8 9
```
---

