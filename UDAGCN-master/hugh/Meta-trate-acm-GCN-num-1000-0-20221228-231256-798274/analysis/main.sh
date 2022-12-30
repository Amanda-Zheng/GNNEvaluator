echo "Start creating data visualization graphs..."

MAIN_LOCATION="${PWD}"
# abstract the useful data from ../test.log
grep -e ".*"-method: -A 1 ../test.log >data/format_data/abstract_main_data.log

# reformat the abstracted data
cd "${MAIN_LOCATION}/data/format_data"
python3 reformat_main_data.py

# filter/group the abstracted data
cd "${MAIN_LOCATION}/data/methods"
python3 filter_methods.py

# sort the abstracted data
cd "${MAIN_LOCATION}/data"
source sort.sh

# create comparison between "ACC" and "FEAT" in each method's visualization graph
cd "${MAIN_LOCATION}/data_visualization"
python3 generate_methods_visualization.py

# create create comparison between ".*rate" (e.g. node_drop_rate...) value and "target" value (e.g. ACC...) in each method's visualization graph
cd "${MAIN_LOCATION}/data_visualization"
python3 generate_field_rate_with_target_field_visualization.py

echo "Done :)"
