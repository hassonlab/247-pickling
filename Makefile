# Instructions
# ------------
# The first time you should run link-data

# For podcast to create a 777 pickle with all electrodes (signal and label)
# glove and gpt2 embeddings:
#   1. create-sig-pickle for 777
#   2. create-pickle for 661
#   3. generate-embeddings for 661 glove and gpt2
#   4. run concatenate-embeddings for 661 glove and gpt2
#   5. run copy-embeddings
#   6. (optionally) run upload-pickle

# For 247
#   1. create-pickle for subject
#   2. generate-embeddings for glove
#   3. generate-embeddings for gpt2
#   4. upload pickle

# Miscellaneous: \
247 Subjects IDs: 625 and 676 \
Podcast Subjects: 661 662 717 723 741 742 743 763 798 \
777: Is the code the collection of significant electrodes

# NOTE: link data from tigressdata before running any scripts
PRJCT_ID := tfs
# {tfs | podcast}

link-data:
ifeq ($(PRJCT_ID), podcast)
	$(eval DIR_KEY := podcast-data)
else
	$(eval DIR_KEY := conversations-car)
endif
	# create directory
	mkdir -p data/$(PRJCT_ID)
	# delete old symlinks
	find data/$(PRJCT_ID)/ -xtype l -delete
	# create symlinks from original data store
	ln -sf /projects/HASSON/247/data/$(DIR_KEY)/* data/$(PRJCT_ID)/

# settings for target: create-pickle, create-sig-pickle, upload-pickle
%-pickle: CMD := python
# {echo | python}
%-pickle: PRJCT_ID := podcast
# {tfs | podcast}
%-pickle: SID_LIST = 777
# {625 676 | 661 662 717 723 741 742 743 763 798 | 777}

create-pickle:
	mkdir -p logs
	for sid in $(SID_LIST); do \
		$(CMD) code/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--subject $$sid; \
	done

# create pickle of significant electrodes (just for podcast)
create-sig-pickle:
	mkdir -p logs
	$(CMD) code/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--sig-elec-file data/$(PRJCT_ID)/all-electrodes.csv

# upload pickles to google cloud bucket
# on bucket we use 247 not tfs, so manually adjust as needed
# upload-pickle: pid=247
upload-pickle: pid=podcast
upload-pickle:
	for sid in $(SID_LIST); do \
		gsutil -m rsync results/$(PRJCT_ID)/$$sid/pickles/ gs://247-podcast-data/$(pid)-pickles/$$sid; \
	done

# upload raw data to google cloud bucket
upload-data:
	gsutil -m rsync -rd data/tfs/676 gs://247-podcast-data/247-data/676/
	gsutil -m rsync -rd data/tfs/625 gs://247-podcast-data/247-data/625/

# download pickles from google cloud bucket
download-247-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247-pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247-pickles/ results/676/


## settings for targets: generate-embeddings, concatenate-embeddings
%-embeddings: CMD := sbatch submit.sh
# {echo | python | sbatch submit.sh}
%-embeddings: PRJCT_ID := tfs
# {tfs | podcast}
%-embeddings: SID := 625
# {625 | 676 | 661} 
%-embeddings: CONV_IDS = $(shell seq 1 54)
# {54 for 625 | 78 for 676 | 1 for 661}
%-embeddings: PKL_IDENTIFIER := full
# {full | trimmed | binned}
%-embeddings: EMB_TYPE := blenderbot-small
# {glove50 | bert | gpt2-xl | gpt2 | gpt2-large }
%-embeddings: CNXT_LEN := 1024
%-embeddings: HIST := --history
# %-embeddings: LAYER := --layer-idx $(shell seq 1 12)
# {48 | 12 for gpt2 | 36 for gpt2-large | 48 for gpt2-xl }
# Note: embeddings file is the same for all podcast subjects \
and hence only generate once using subject: 661

# 38 and 39 failed

# generates embeddings (for each conversation separately)
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
			$(LAYER) \
			--context-length $(CNXT_LEN); \
	done;

# concatenate embeddings from all conversations
concatenate-embeddings:
	python code/tfsemb_concat.py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--subject $(SID) \
		--embedding-type $(EMB_TYPE) \
		$(HIST) \
		--context-length $(CNXT_LEN); \

# Podcast: copy embeddings to other subjects as well
# for sid in 662 717 723 741 742 763 798 777; do 
copy-embeddings:
	@for fn in results/podcast/661/pickles/*embeddings.pkl; do \
		for sid in 777; do \
			cp -pf $$fn $$(echo $$fn | sed "s/661/$$sid/g"); \
		done; \
	done
