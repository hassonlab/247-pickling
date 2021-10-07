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

# Miscellaneous: \
247 Subjects IDs: 625 and 676 \
Podcast Subjects: 661 662 717 723 741 742 743 763 798 \
777: Is the code the collection of significant electrodes

# NOTE: link data from tigressdata before running any scripts
PRJCT_ID := podcast
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

# create pickle fof significant electrodes
create-sig-pickle:
	mkdir -p logs
	$(CMD) code/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--sig-elec-file data/$(PRJCT_ID)/all-electrodes.csv

# upload pickles to google cloud bucket
upload-pickle:
	for sid in $(SID_LIST); do \
		gsutil -m rsync results/$(PRJCT_ID)/$$sid/pickles/ gs://247-podcast-data/$(PRJCT_ID)-pickles/$$sid; \
	done

# download pickles from google cloud bucket
download-247-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247_pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247_pickles/ results/676/


## settings for targets: generate-embeddings, concatenate-embeddings
%-embeddings: CMD := python
# {echo | python | sbatch submit.sh}
%-embeddings: PRJCT_ID := podcast
# {tfs | podcast}
%-embeddings: SID := 661
# {625 | 676 | 661} 
%-embeddings: CONV_IDS = $(shell seq 1 1)
# {54 for 625 | 79 for 676 | 1 for 661}
%-embeddings: PKL_IDENTIFIER := full
# {full | trimmed | binned}
%-embeddings: EMB_TYPE := gpt2-xl
# {glove50 | bert | gpt2-xl}
%-embeddings: CNXT_LEN := 1024
%-embeddings: HIST := --history
%-embeddings: LAYER := 48
# Note: embeddings file is the same for all podcast subjects \
and hence only generate once using subject: 661

# generates embeddings (for each conversation separately)
generate-embeddings:
	mkdir -p logs
	for layer in $(LAYER); do \
		for conv_id in $(CONV_IDS); do \
			$(CMD) code/tfsemb_main.py \
						--project-id $(PRJCT_ID) \
						--pkl-identifier $(PKL_IDENTIFIER) \
						--subject $(SID) \
						--conversation-id $$conv_id \
						--embedding-type $(EMB_TYPE) \
						$(HIST) \
						--context-length $(CNXT_LEN) \
						--layer-idx $$layer; \
		done; \
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
