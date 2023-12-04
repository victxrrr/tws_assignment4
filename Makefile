.PHONY: plots

simulation1:
	g++ -Wall -std=c++17 -O3 -o simulation1.out src/simulation1.cpp

simulation2:
	g++ -Wall -std=c++17 -O3 -o simulation2.out src/simulation2.cpp

estimation1:
	# Put your compile command for estimation1.cpp here. The name of your executable should be estimation1

estimation2:
	# Put your compile command for estimation2.cpp here. The name of your executable should be estimation2

plots:
	cd ./plots && pdflatex plot.tex >/dev/null && cd ..

clean:
	rm -f *.out