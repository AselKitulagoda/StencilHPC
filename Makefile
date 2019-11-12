
stencil: stencil.c
	mpiicc -std=c99 -g -Ofast -march=native -Wall $^ -o $@


