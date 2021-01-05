# create 'data' and 'results' folders (note: case sensitive)
mkdir -p data results

# delete bad symlinks
find data/ -xtype l -delete

# rm data/*

# create symlinks from original data store
ln -s /projects/HASSON/247/data/conversations-car/* data/
