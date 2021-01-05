# create 'data' and 'results' folders (note: case sensitive)
mkdir -p data results

cd data

# delete bad symlinks
find . -xtype l -delete

cd ..

# rm data/*

# create symlinks from original data store
ln -s /projects/HASSON/247/data/conversations-car/* data/
