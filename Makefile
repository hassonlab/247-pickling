CMD := echo

# 247 subjects
PRJCT_ID := tfs
SID_LIST=625

# For 625
SID := 625
CONV_IDS = $(shell seq 1 54)

# For 676
SID := 676
CONV_IDS = $(shell seq 1 79)

# # # podcast subjects
PRJCT_ID := podcast
SID_LIST=661 662 717 723 741 742 763 798
SID_LIST=777

EMB_TYPE := glove50
EMB_TYPE := bert
EMB_TYPE := gpt2-xl

PKL_IDENTIFIER := full
PKL_IDENTIFIER := trimmed
PKL_IDENTIFIER := binned

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

create-pickle:
	mkdir -p logs
	for sid in $(SID_LIST); do \
		python code/tfspkl_main.py \
					--project-id $(PRJCT_ID) \
					--subject $$sid \
					--max-electrodes $(MEL) \
					--vocab-min-freq $(MINF); \
		done

create-sig-pickle:
	mkdir -p logs
	python code/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--sig-elec-file /scratch/gpfs/hgazula/phase-5000-sig-elec-glove50d-perElec-FDR-01_newVer_1000perm-LH.csv \
			--max-electrodes $(MEL) \
			--vocab-min-freq $(MINF);

upload-pickle:
	for sid in $(SID_LIST); do \
		gsutil -m cp -r results/$(PRJCT_ID)/$$sid/pickles/*.pkl gs://247-podcast-data/$(PRJCT_ID)_pickles/$$sid; \
	done

download-247-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247_pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247_pickles/ results/676/

CMD := sbatch submit.sh
generate-embeddings:
	mkdir -p logs
	for conv_id in $(CONV_IDS); do \
		$(CMD) code/tfsemb_main.py \
					--project-id $(PRJCT_ID) \
					--pkl-identifier $(PKL_IDENTIFIER) \
					--subject $(SID) \
					--conversation-id $$conv_id \
					--embedding-type $(EMB_TYPE) \
					$(HIST) \
					--context-length $(CNXT_LEN); \
	done

concatenate-embeddings:
	python code/tfsemb_concat.py \
				--project-id $(PRJCT_ID) \
				--pkl-identifier $(PKL_IDENTIFIER) \
				--subject $(SID) \
				--embedding-type $(EMB_TYPE) \
				$(HIST) \
				--context-length $(CNXT_LEN); \

# Sync results with the /projects/HASSON folder
sync-results:
	rsync -ah /scratch/gpfs/$(shell whoami)/247-pickling/results/* /projects/HASSON/247/results_new_infra/pickling/
