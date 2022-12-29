DELIMITER=','
ACC=6

: '
sort whole data
'
sort -t $DELIMITER -k $ACC 'format_data/reformatted_main_data.log' >'format_data/sorted_reformatted_main_data.log'

:'
sort data in each method
'
SORTED_FILE_PREFIX='methods/sorted'
mkdir $SORTED_FILE_PREFIX
for file in methods/*.log; do
  fileName="${file##*/}"
  newFileLocation="$SORTED_FILE_PREFIX/$fileName"
  sort -t $DELIMITER -k $ACC methods/$fileName > $newFileLocation
done
