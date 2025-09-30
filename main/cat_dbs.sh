#!/bin/bash

outfile="db.csv"

dbs=(db-*.csv)
echo Copying "${dbs[0]}" to "${outfile}"
cp "${dbs[0]}" "${outfile}"

for file in "${dbs[@]:1}"; do  # loop over all files after the first one
	echo Adding "${file}" to "${outfile}"
	# skip the first line (header) of subsequent files
	sed 1,1d "${file}" >> "${outfile}"
done
