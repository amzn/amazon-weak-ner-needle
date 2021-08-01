#!/bin/sh

URL=ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline
for i in $(seq -f "%04g" 1 1015); do
  GZFILE=pubmed20n${i}.xml.gz
  echo $URL/$GZFILE
  wget $URL/$GZFILE
  gzip -d $GZFILE
  XMLFILE=pubmed20n${i}.xml
  TGTFILE=${i}.txt
  cat $XMLFILE | grep '<ArticleTitle>\|<AbstractText>' \
    |  awk '{gsub(" *</?ArticleTitle>","",$0); gsub(" *</?AbstractText>","",$0); print$0}' \
    > $TGTFILE
  rm $XMLFILE
done
