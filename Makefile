# Miscellaneous: \
247 Subjects IDs: 625 and 676 \
Podcast Subjects: 661 662 717 723 741 742 743 763 798 \
777: Is the code the collection of significant electrodes

# NOTE: link data from tigressdata before running any scripts \
(Recommend running this before running every target)
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

	# delete bad symlinks
	find data/$(PRJCT_ID)/ -xtype l -delete

	# create symlinks from original data store
	ln -sf /projects/HASSON/247/data/$(DIR_KEY)/* data/$(PRJCT_ID)/


# settings for target: create-pickle, create-sig-pickle, upload-pickle
%-pickle: CMD := python 		# {echo | python}
%-pickle: PRJCT_ID := podcast 	# {tfs | podcast}
%-pickle: SID_LIST=743 		# {625 676 | 661 662 717 723 741 742 743 763 798 | 777}
%-pickle: MEL := 500 		# Setting a large number will extract all common \
									electrodes across all conversations
%-pickle: MINF := 0

create-pickle:
	mkdir -p logs
	for sid in $(SID_LIST); do \
		$(CMD) code/tfspkl_main.py \
					--project-id $(PRJCT_ID) \
					--subject $$sid \
					--max-electrodes $(MEL) \
					--vocab-min-freq $(MINF); \
		done

# create pickle fof significant electrodes
create-sig-pickle:
	mkdir -p logs
	$(CMD) code/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--sig-elec-file /scratch/gpfs/hgazula/phase-5000-sig-elec-glove50d-perElec-FDR-01_newVer_1000perm-LH.csv \
			--max-electrodes $(MEL) \
			--vocab-min-freq $(MINF);

# upload pickles to google cloud bucket
upload-pickle:
	for sid in $(SID_LIST); do \
		gsutil -m cp -r results/$(PRJCT_ID)/$$sid/pickles/*.pkl gs://247-podcast-data/$(PRJCT_ID)_pickles/$$sid; \
	done

# download pickles from google cloud bucket
download-247-pickles:
	mkdir -p results/{625,676}
	gsutil -m rsync -x "^(?!.*625).*" gs://247-podcast-data/247_pickles/ results/625/
	gsutil -m rsync -x "^(?!.*676).*" gs://247-podcast-data/247_pickles/ results/676/


## settings for targets: generate-embeddings, concatenate-embeddings
%-embeddings: CMD := sbatch submit.sh  	# {echo | python | sbatch submit.sh}
%-embeddings: PRJCT_ID := podcast  		# {tfs | podcast}
%-embeddings: SID := 661 				# {625 | 676 | 661} 
	
%-embeddings: CONV_IDS = $(shell seq 1 1) # {54 for 625 | 79 for 676 | 1 for 661}
%-embeddings: PKL_IDENTIFIER := full 	# {full | trimmed | binned}
%-embeddings: EMB_TYPE := gpt2-xl 		# {glove50 | bert | gpt2-xl}
%-embeddings: CNXT_LEN := 1024
%-embeddings: HIST := --history
# Note: embeddings file is the same for all podcast subjects \
and hence only generate once using subject: 661

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
					--context-length $(CNXT_LEN); \
	done

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
copy-embeddings:
	for SID_LIST=662 717 723 741 742 763 798 | 777}; do \
		rsync /scratch/gpfs/$(shell whoami)/247-pickling/results/podcast/661/
	done

# Sync results with the /projects/HASSON folder
sync-results:
	rsync -ah /scratch/gpfs/$(shell whoami)/247-pickling/results/* /projects/HASSON/247/results_new_infra/pickling/
