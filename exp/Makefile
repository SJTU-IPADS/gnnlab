run_list_1=fig4a.run fig4b.run fig5a.run fig5b.run fig10.run fig11a.run fig11b.run fig11c.run fig12.run table2.run
run_list_2=table1.run table4.run table5.run fig13a.run fig13b.run fig15.run
run_list_3=fig16a.run fig17a.run fig17b.run
clean_list_1=$(patsubst %.run,%.clean,$(run_list_1))

all: $(run_list_1) $(run_list_2) $(run_list_3)

# target looks like "fig4a.run", and "$(patsubst %.run,%,$@)" gives "fig4a"
$(run_list_1):
	cd $(patsubst %.run,%,$@); if [ ! -e "run-logs/run.fin" ]; then python3 runner.py && touch run-logs/run.fin ; fi
	cd $(patsubst %.run,%,$@); python3 parser.py; if [ -e "plot.plt" ]; then gnuplot plot.plt ; fi

$(run_list_2):
	cd $(patsubst %.run,%,$@); python3 run.py

$(run_list_3):
	cd $(patsubst %.run,%,$@); bash run.sh

$(clean_list_1):
	rm -rf $(patsubst %.clean,%,$@)/run-logs

clean: $(clean_list_1)

.PHONY: clean
