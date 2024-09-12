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
247 Subjects IDs: 625, 676, 7170, and 798 \
Podcast Subjects: 661 662 717 723 741 742 743 763 798 \
777: Is the collection of significant electrodes

# Other Notes:
# For generate-embeddings
# 1. Models greater than 6.7B parameters need a GPU with more than 40GB RAM
# 2. This means currently we cannot run EleutherAI/gpt-neox-20b and 
# facebook/opt-30b using huggingface on della GPUs

# Run all commands in one shell
.ONESHELL:

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
%-pickle: PRJCT_ID := tfs
# {tfs | podcast}
%-pickle: SID_LIST = 625
# {625 676 7170 798 | 661 662 717 723 741 742 743 763 798 | 777}

create-pickle:
	mkdir -p logs
	for sid in $(SID_LIST); do \
		$(CMD) scripts/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--subject $$sid; \
	done

# create pickle of significant electrodes (just for podcast)
create-sig-pickle:
	mkdir -p logs
	$(CMD) scripts/tfspkl_main.py \
			--project-id $(PRJCT_ID) \
			--sig-elec-file all-electrodes2.csv


##############################################
############ Embedding Generation ############
##############################################

SID := 625
# {625 | 676 | 7170 | 798 | 661} 
CONV_IDS := $(shell seq 1 54)
CONV_IDS := $(shell seq 1 1)
# {54 for 625 | 78 for 676 | 1 for 661 | 24 for 7170 | 15 for 798}

JOB_NAME := $(subst /,-,$(EMB_TYPE))
CMD := sbatch --job-name=$(SID)-$(JOB_NAME)-cnxt-$$cnxt_len-conv-$$conv_id submit.sh
CMD := python
# {echo | python | sbatch --job-name=$(SID)-$(JOB_NAME)-cnxt-$$cnxt_len submit.sh}


# generate-base-for-embeddings: Generates the base dataframe for embedding generation
generate-base:
	python scripts/tfsemb_LMBase.py \
		--config-file $(SID).yml glove.yml

generate-embeddings:
	mkdir -p logs
	for conv_id in $(CONV_IDS); do \
		$(CMD) scripts/tfsemb_main.py \
			--config-file $(SID).yml gpt2.yml \
			--conv-id $$conv_id; \
	done;



# generates embeddings (for each conversation separately)
# generate-embeddings:
# 	mkdir -p logs
# 	for cnxt_len in $(CNXT_LEN); do \
# 		for conv_id in $(CONV_IDS); do \
# 			$(CMD) scripts/tfsemb_main.py \
# 				--project-id $(PRJCT_ID) \
# 				--pkl-identifier $(PKL_IDENTIFIER) \
# 				--subject $(SID) \
# 				--conversation-id $$conv_id \
# 				--embedding-type $(EMB_TYPE) \
# 				--model-type $(MDL_TYPE) \
# 				--shuffle-audio $(SHUFFLE_AUDIO) \
# 				--shuffle-words $(SHUFFLE_WORDS) \
# 				--cutoff $(CUTOFF) \
# 				$(PROD_COMP_SPLIT) \
# 				$(MASKED) \
# 				$(LCTX) \
# 				$(RCTX) \
# 				$(RCTXP) \
# 				$(UTT) \
# 				--layer-idx $(LAYER) \
# 				--context-length $$cnxt_len; \
# 		done; \
# 	done;

# concatenate embeddings from all conversations
concatenate-embeddings:
	for cnxt_len in $(CNXT_LEN); do \
		$(CMD) scripts/tfsemb_concat.py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER) \
			--subject $(SID) \
			--embedding-type $(EMB_TYPE) \
			--context-length $$cnxt_len; \
	done;

# Podcast: copy embeddings to other subjects as well
copy-embeddings:
	fn=results/podcast/661/pickles/embeddings
	for sid in 662 717 723 741 742 763 798 777; do \
		cp -rpf $$fn $$(echo $$fn | sed "s/661/$$sid/g"); \
	done; \

perp-embeddings:
	mkdir -p logs
	for conv_id in $(CONV_IDS); do \
		$(CMD) scripts/tfsemb_perplexity.py \
			--project-id $(PRJCT_ID) \
			--pkl-identifier $(PKL_IDENTIFIER) \
			--subject $(SID) \
			--conversation-id $$conv_id \
			--embedding-type $(EMB_TYPE); \
	done;



# Download huggingface models to cache (before generating embeddings)
# This target needs to be run on the head node
cache-models: MODEL := openai/whisper-tiny.en
# {causal | seq2seq | mlm | or any model name specified in EMB_TYPE comments}
cache-models:
	python -c "from scripts import tfsemb_download; tfsemb_download.download_tokenizers_and_models(\"$(MODEL)\")"

# perp-embeddings: Generate perplexity scores (for more info see issue170)
perp-embeddings:
	mkdir -p logs
	for conv_id in $(CONV_IDS); do \
		python scripts/tfsemb_perplexity.py \
		--project-id $(PRJCT_ID) \
		--pkl-identifier $(PKL_IDENTIFIER) \
		--subject $(SID) \
		--conversation-id $$conv_id \
		--embedding-type $(EMB_TYPE); \
	done;