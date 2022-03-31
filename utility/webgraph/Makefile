
ifeq ($(mvn), )
mvn=mvn
endif
build:
	$(mvn) clean package
convert:
ifeq ($(graph),)
	@echo "please set graph name!"
else
	java -cp target/webgraph-0.1-SNAPSHOT.jar it.unimi.dsi.webgraph.BVGraph -o -O -L $(graph)
	java -cp target/webgraph-0.1-SNAPSHOT.jar ipads.samgraph.webgraph.WebgraphDecoder $(graph)
endif
help:
	@echo "Usage: make build; make convert graph=/graph-learning/data-raw/example"
	@echo "    there must exist file 'example.graph' and 'example.properties'"