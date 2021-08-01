mkdir tasks
cd tasks
git clone https://github.com/cambridgeltl/MTL-Bioinformatics-2016.git
cp -r MTL-Bioinformatics-2016/data/BC5CDR-chem-IOB ./BC5CDR-chem
cp -r MTL-Bioinformatics-2016/data/BC5CDR-disease-IOB ./BC5CDR-disease
cp -r MTL-Bioinformatics-2016/data/NCBI-disease-IOB ./NCBI-disease
rm -rf MTL-Bioinformatics-2016

mv BC5CDR-chem/devel.tsv BC5CDR-chem/dev.txt
mv BC5CDR-chem/train.tsv BC5CDR-chem/train.txt
mv BC5CDR-chem/test.tsv BC5CDR-chem/test.txt
mv BC5CDR-disease/devel.tsv BC5CDR-disease/dev.txt
mv BC5CDR-disease/train.tsv BC5CDR-disease/train.txt
mv BC5CDR-disease/test.tsv BC5CDR-disease/test.txt
mv NCBI-disease/devel.tsv NCBI-disease/dev.txt
mv NCBI-disease/train.tsv NCBI-disease/train.txt
mv NCBI-disease/test.tsv NCBI-disease/test.txt

echo "O" >> chem_labels.txt
echo "B-Chemical" >> chem_labels.txt
echo "I-Chemical" >> chem_labels.txt
mv chem_labels.txt BC5CDR-chem/labels.txt 

echo "O" >> disease_labels.txt
echo "B-Disease" >> disease_labels.txt
echo "I-Disease" >> disease_labels.txt
cp disease_labels.txt NCBI-disease/labels.txt
mv disease_labels.txt BC5CDR-disease/labels.txt