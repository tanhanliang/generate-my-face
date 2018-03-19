i=0
for fname in ./*; do
    mv "$fname" "real_$i.jpg"
    i=$((i+1))
done
