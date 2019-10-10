
stencil: stencil.c
	gcc -std=c99 -pg -Ofast -Wall $^ -o $@


