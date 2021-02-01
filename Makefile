CMD := echo
CMD := python
CMD := sbatch submit1.sh

# For 625
CONV_IDS = $(shell seq 1 54)

# For 676
CONV_IDS = $(shell seq 1 79)

EMB_TYPE := glove50
EMB_TYPE := bert
EMB_TYPE := gpt2-xl

SID := 625
SID := 676

# a very large number for MEL will extract all common...
# ...electrodes across all conversations
MEL := 500
MINF := 30

HIST := --history
CNXT_LEN := 1024


link-data:
	# delete bad symlinks
	find data/ -xtype l -delete

	# create symlinks from original data store
	ln -sf /projects/HASSON/247/data/conversations-car/* data/

download-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247_pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247_pickles/ results/676/

create-pickle: link-data
	mkdir -p logs
	$(CMD) code/tfspkl_main.py \
				--subject $(SID) \
				--max-electrodes $(MEL) \
				--vocab-min-freq $(MINF);

upload-pickle: create-pickle
	gsutil -m cp -r results/$(SID)/$(SID)*.pkl gs://247-podcast-data/247_pickles/

generate-embeddings: link-data
	mkdir -p logs
	for conv_id in $(CONV_IDS); do \
		$(CMD) code/tfsemb_main.py \
					--subject $(SID) \
					--conversation-id $$conv_id \
					--embedding-type $(EMB_TYPE) \
					$(HIST) \
					--context-length $(CNXT_LEN); \
	done

concatenate-embeddings:
	python code/tfsemb_concat.py \
					--subject $(SID) \
					--embedding-type $(EMB_TYPE) \
					$(HIST) \
					--context-length $(CNXT_LEN); \
