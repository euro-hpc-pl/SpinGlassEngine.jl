for size in 2 3 4
do
	for cat in RAU RCO AC3
  	do
    	cd "Z$size"
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
