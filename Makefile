# US5 / TC-05: every published number regenerates from one command.
BEIR_BASE := https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets
DATASETS := scifact nfcorpus
DATA := benchmarks/data

.PHONY: bench bench-data

bench: bench-data
	python -m benchmarks.harness

bench-data:
	@mkdir -p $(DATA)
	@for d in $(DATASETS); do \
	  if [ ! -d $(DATA)/$$d ]; then \
	    echo "fetching $$d (pinned BEIR bundle)"; \
	    curl -sL "$(BEIR_BASE)/$$d.zip" -o $(DATA)/$$d.zip; \
	    echo "$$(grep "^$$d " benchmarks/PINS.sha256 | cut -d' ' -f2)  $(DATA)/$$d.zip" | shasum -a 256 -c - \
	      || { echo "CHECKSUM MISMATCH for $$d — refusing to benchmark unpinned data"; rm $(DATA)/$$d.zip; exit 1; }; \
	    unzip -q $(DATA)/$$d.zip -d $(DATA); \
	    rm $(DATA)/$$d.zip; \
	  fi; \
	done
