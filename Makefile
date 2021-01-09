CMD := echo
CMD := python
# CMD := sbatch submit1.sh

EMB_TYPE := glove50
EMB_TYPE := bert
EMB_TYPE := gpt2

SID := 625
# SID := 676

# a very large number for MEL will extract all common...
# ...electrodes across all conversations
MEL := 500
MINF := 30

HIST := --history
CNXT_LEN := 0


link-data:
	# delete bad symlinks
	find data/ -xtype l -delete

	# create symlinks from original data store
	ln -sf /projects/HASSON/247/data/conversations-car/* data/


create-pickle:
	mkdir -p logs
	$(CMD) code/tfs_pickling.py \
				--subjects $(SID) \
				--max-electrodes $(MEL) \
				--vocab-min-freq $(MINF) \
				--pickle;

upload-pickle: create-pickle
	gsutil -m cp -r results/$(SID)/$(SID)*.pkl gs://247-podcast-data/247_pickles/

generate-embeddings:
	mkdir -p logs
	$(CMD) code/tfs_gen_embeddings.py \
				--subject $(SID) \
				--embedding-type $(EMB_TYPE) \
				$(HIST) \
				--context-length $(CNXT_LEN);
