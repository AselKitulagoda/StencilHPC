
stencil: stencil.c
	icc -std=c99 -fast -march=native -Wall $^ -o $@


