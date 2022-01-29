run_list_1=fig4a.run fig4b.run fig5a.run fig5b.run fig10.run fig11a.run fig11b.run fig11c.run fig12.run table2.run
clean_list_1=$(patsubst %.run,%.clean,$(run_list_1))

# target looks like "fig4a.run", and "$(patsubst %.run,%,$@)" gives "fig4a"
$(run_list_1):
	cd $(patsubst %.run,%,$@); if [ ! -e "run-logs/run.fin" ]; then python3 runner.py && touch run-logs/run.fin ; fi
	cd $(patsubst %.run,%,$@); python3 parser.py; if [ -e "plot.plt" ]; then gnuplot plot.plt ; fi

$(clean_list_1):
	rm -rf $(patsubst %.clean,%,$@)/run-logs

all: $(run_list_1)

clean: $(clean_list_1)

.PHONY: clean