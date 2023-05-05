for size in 12 16
do
  for cat in CBFM-P
  do
    cd "P$size"
    cd "$cat"
	mkdir SpinGlass
	mkdir DWave
	for f in ./*.txt
	do
		mv $f SpinGlass
	done
	for f in ./*.pkl
	do
		mv $f DWave
	done

	cd ..
	cd ..
  done
done
