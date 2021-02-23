CMD := echo

# For 625
CONV_IDS = $(shell seq 1 54)

# For 676
CONV_IDS = $(shell seq 1 79)

EMB_TYPE := glove50
EMB_TYPE := bert
EMB_TYPE := gpt2-xl

PRJCT_ID := podcast
PRJCT_ID := tfs

# 247 subjects
SID_LIST=625 676

# podcast subjects
SID_LIST=661 662 717 723 741 742 763 798 

# a very large number for MEL will extract all common...
# ...electrodes across all conversations
MEL := 500
MINF := 0

HIST := --history
CNXT_LEN := 1024

link-data:
ifeq ($(PRJCT_ID), podcast)
	$(eval DIR_KEY := podcast-data)
else
	$(eval DIR_KEY := conversations-car)
endif

	# create directory
	mkdir -p data/$(PRJCT_ID)

	# delete bad symlinks
	find data/$(PRJCT_ID)/ -xtype l -delete

	# create symlinks from original data store
	ln -sf /projects/HASSON/247/data/$(DIR_KEY)/* data/$(PRJCT_ID)/


download-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247_pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247_pickles/ results/676/

CMD := python
create-pickle:
	mkdir -p logs
	for sid in $(SID_LIST); do \
		$(CMD) code/tfspkl_main.py \
					--project-id $(PRJCT_ID) \
					--subject $$sid \
					--max-electrodes $(MEL) \
					--vocab-min-freq $(MINF); \
		done

upload-247-pickle: create-pickle
	gsutil -m cp -r results/$(SID)/pickles/$(SID)*.pkl gs://247-podcast-data/247_pickles/

upload-podcast-pickle: create-pickle
	for sid in $(SID_LIST); do \
		gsutil -m cp -r results/$$sid/pickles/*.pkl gs://247-podcast-data/podcast_pickles/$$sid; \
	done

download-247-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247_pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247_pickles/ results/67

# CMD := sbatch submit1.sh
generate-embeddings:
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
