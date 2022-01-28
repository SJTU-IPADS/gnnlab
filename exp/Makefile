plotfiles=$(wildcard *.plot)
svgfiles = $(patsubst %.plot,%.svg,$(plotfiles))
pdffiles = $(patsubst %.plot,%.pdf,$(plotfiles))

sxn_tests=fig4a.test fig4b.test fig5a.test fig5b.test fig10.test fig11a.test fig11b.test fig11c.test fig12.test table2.test
sxn_run=$(patsubst %.test,%/run-logs/run.fin,$(sxn_tests))
sxn_parse=$(patsubst %.test,%/run-logs/parse.fin,$(sxn_tests))

%.test: %/run-logs/parse.fin

%/run-logs/run.fin:
	rm -rf $*/run-logs
	cd $*; python3 runner.py
	touch $*/run-logs/run.fin

%/run-logs/parse.fin: %/run-logs/run.fin
	rm -f $*/run-logs/parse.fin
	cd $*; python3 parser.py
	cd $*; if [ -e "plot.plt" ]; then  gnuplot plot.plt && ps2pdf -dEPSCrop $*.eps $*.pdf; fi
	touch $*/run-logs/parse.fin

%.clean:
	rm -rf $*/run-logs

all: $(sxn_tests)

run: $(sxn_run)

parse: $(sxn_parse)

plot: $()

.PHONY: $(sxn_tests)