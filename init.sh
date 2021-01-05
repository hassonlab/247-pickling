mkdir -p data results

cd data
find . -xtype l -delete

cd ..
# rm data/*
ln -s /projects/HASSON/247/data/conversations-car/* data/

